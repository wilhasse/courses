package byodb08

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path"
	"syscall"

	"golang.org/x/sys/unix"
)

type KV struct {
	Path   string
	NoSync bool // for testing
	// internals
	fd   int
	tree BTree
	free FreeList
	mmap struct {
		total  int      // mmap size, can be larger than the file size
		chunks [][]byte // multiple mmaps, can be non-continuous
	}
	page struct {
		flushed uint64            // database size in number of pages
		nappend uint64            // number of pages to be appended
		updates map[uint64][]byte // pending updates, including appended pages
	}
}

// `BTree.get`, read a page.
func (db *KV) pageRead(ptr uint64) []byte {
	assert(ptr < db.page.flushed+db.page.nappend)
	if node, ok := db.page.updates[ptr]; ok {
		return node // pending update
	}
	return db.pageReadFile(ptr)
}

func (db *KV) pageReadFile(ptr uint64) []byte {
	start := uint64(0)
	for _, chunk := range db.mmap.chunks {
		end := start + uint64(len(chunk))/BTREE_PAGE_SIZE
		if ptr < end {
			offset := BTREE_PAGE_SIZE * (ptr - start)
			return chunk[offset : offset+BTREE_PAGE_SIZE]
		}
		start = end
	}
	panic("bad ptr")
}

// `BTree.new`, allocate a new page.
func (db *KV) pageAlloc(node []byte) uint64 {
	assert(len(node) == BTREE_PAGE_SIZE)
	if ptr := db.free.PopHead(); ptr != 0 { // try the free list
		assert(db.page.updates[ptr] == nil)
		db.page.updates[ptr] = node
		return ptr
	}
	return db.pageAppend(node) // append
}

// `FreeList.new`, append a new page.
func (db *KV) pageAppend(node []byte) uint64 {
	assert(len(node) == BTREE_PAGE_SIZE)
	ptr := db.page.flushed + db.page.nappend
	db.page.nappend++
	assert(db.page.updates[ptr] == nil)
	db.page.updates[ptr] = node
	return ptr
}

// `FreeList.set`, update an existing page.
func (db *KV) pageWrite(ptr uint64) []byte {
	assert(ptr < db.page.flushed+db.page.nappend)
	if node, ok := db.page.updates[ptr]; ok {
		return node // pending update
	}
	node := make([]byte, BTREE_PAGE_SIZE)
	copy(node, db.pageReadFile(ptr)) // initialized from the file
	assert(db.page.updates[ptr] == nil)
	db.page.updates[ptr] = node
	return node
}

// create the initial mmap that covers the whole file.
func initMmap(fd int, fileSize int64) ([]byte, error) {
	mmapSize := 64 << 20
	for mmapSize < int(fileSize) {
		mmapSize *= 2 // can be larger than the file
	}
	chunk, err := syscall.Mmap(
		fd, 0, mmapSize, syscall.PROT_READ, syscall.MAP_SHARED,
	)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}
	return chunk, nil
}

// open or create a file and fsync the directory
func createFileSync(file string) (int, error) {
	// obtain the directory fd
	flags := os.O_RDONLY | syscall.O_DIRECTORY
	dirfd, err := syscall.Open(path.Dir(file), flags, 0o644)
	if err != nil {
		return -1, fmt.Errorf("open directory: %w", err)
	}
	defer syscall.Close(dirfd)
	// open or create the file
	flags = os.O_RDWR | os.O_CREATE
	fd, err := syscall.Openat(dirfd, path.Base(file), flags, 0o644)
	if err != nil {
		return -1, fmt.Errorf("open file: %w", err)
	}
	// fsync the directory
	err = syscall.Fsync(dirfd)
	if err != nil { // may leave an empty file
		_ = syscall.Close(fd)
		return -1, fmt.Errorf("fsync directory: %w", err)
	}
	// done
	return fd, nil
}

// open or create a DB file
func (db *KV) Open() error {
	var err error
	var chunk []byte
	db.page.updates = map[uint64][]byte{}
	// B+tree callbacks
	db.tree.get = db.pageRead
	db.tree.new = db.pageAlloc
	db.tree.del = db.free.PushTail
	// free list callbacks
	db.free.get = db.pageRead
	db.free.new = db.pageAppend
	db.free.set = db.pageWrite
	// open or create the DB file
	if db.fd, err = createFileSync(db.Path); err != nil {
		return err
	}
	// get the file size
	finfo := syscall.Stat_t{}
	if err = syscall.Fstat(db.fd, &finfo); err != nil {
		goto fail
	}
	// create the initial mmap
	if chunk, err = initMmap(db.fd, finfo.Size); err != nil {
		goto fail
	}
	db.mmap.total = len(chunk)
	db.mmap.chunks = [][]byte{chunk}
	// read the meta page
	if err = readRoot(db, finfo.Size); err != nil {
		goto fail
	}
	return nil
	// error
fail:
	db.Close()
	return fmt.Errorf("KV.Open: %w", err)
}

const DB_SIG = "BuildYourOwnDB08"

/*
the 1st page stores the root pointer and other auxiliary data.
| sig | root_ptr | page_used | head_page | head_seq | tail_page | tail_seq |
| 16B |    8B    |     8B    |     8B    |    8B    |     8B    |    8B    |
*/
func loadMeta(db *KV, data []byte) {
	db.tree.root = binary.LittleEndian.Uint64(data[16:24])
	db.page.flushed = binary.LittleEndian.Uint64(data[24:32])
	db.free.headPage = binary.LittleEndian.Uint64(data[32:40])
	db.free.headSeq = binary.LittleEndian.Uint64(data[40:48])
	db.free.tailPage = binary.LittleEndian.Uint64(data[48:56])
	db.free.tailSeq = binary.LittleEndian.Uint64(data[56:64])
}

func saveMeta(db *KV) []byte {
	var data [64]byte
	copy(data[:16], []byte(DB_SIG))
	binary.LittleEndian.PutUint64(data[16:24], db.tree.root)
	binary.LittleEndian.PutUint64(data[24:32], db.page.flushed)
	binary.LittleEndian.PutUint64(data[32:40], db.free.headPage)
	binary.LittleEndian.PutUint64(data[40:48], db.free.headSeq)
	binary.LittleEndian.PutUint64(data[48:56], db.free.tailPage)
	binary.LittleEndian.PutUint64(data[56:64], db.free.tailSeq)
	return data[:]
}

func readRoot(db *KV, fileSize int64) error {
	if fileSize%BTREE_PAGE_SIZE != 0 {
		return errors.New("file is not a multiple of pages")
	}
	if fileSize == 0 { // empty file
		// reverve 2 pages: the meta page and a free list node
		db.page.flushed = 2
		// add an initial node to the free list so it's never empty
		db.free.headPage = 1 // the 2nd page
		db.free.tailPage = 1
		return nil // the meta page will be written in the 1st update
	}
	// read the page
	data := db.mmap.chunks[0]
	loadMeta(db, data)
	// initialize the free list
	db.free.SetMaxSeq()
	// verify the page
	bad := !bytes.Equal([]byte(DB_SIG), data[:16])
	// pointers are within range?
	maxpages := uint64(fileSize / BTREE_PAGE_SIZE)
	bad = bad || !(0 < db.page.flushed && db.page.flushed <= maxpages)
	bad = bad || !(0 < db.tree.root && db.tree.root < db.page.flushed)
	bad = bad || !(0 < db.free.headPage && db.free.headPage < db.page.flushed)
	bad = bad || !(0 < db.free.tailPage && db.free.tailPage < db.page.flushed)
	if bad {
		return errors.New("bad meta page")
	}
	return nil
}

// update the meta page. it must be atomic.
func updateRoot(db *KV) error {
	// NOTE: atomic?
	if _, err := syscall.Pwrite(db.fd, saveMeta(db), 0); err != nil {
		return fmt.Errorf("write meta page: %w", err)
	}
	return nil
}

// extend the mmap by adding new mappings.
func extendMmap(db *KV, size int) error {
	for db.mmap.total < size {
		// double the address space
		chunk, err := syscall.Mmap(
			db.fd, int64(db.mmap.total), db.mmap.total,
			syscall.PROT_READ, syscall.MAP_SHARED,
		)
		if err != nil {
			return fmt.Errorf("mmap: %w", err)
		}

		db.mmap.total += db.mmap.total
		db.mmap.chunks = append(db.mmap.chunks, chunk)
	}
	return nil
}

func fsyncFile(db *KV) error {
	if db.NoSync { // skip fsync for testing
		return nil
	}
	err := syscall.Fsync(db.fd)
	if err != nil {
		err = fmt.Errorf("fsync: %w", err)
	}
	return err
}

func tryUpdateFile(db *KV) error {
	// 1. Write new nodes.
	if err := writePages(db); err != nil {
		return err
	}
	// 2. `fsync` to enforce the order between 1 and 3.
	if err := fsyncFile(db); err != nil {
		return err
	}
	// 3. Update the root pointer atomically.
	if err := updateRoot(db); err != nil {
		return err
	}
	// 4. `fsync` to make everything persistent.
	if err := fsyncFile(db); err != nil {
		return err
	}
	// prepare the free list for the next update
	db.free.SetMaxSeq()
	return nil
}

func updateFile(db *KV) error {
	// save the in-memory state
	meta := saveMeta(db)
	// update the file
	err := tryUpdateFile(db)
	// revert the in-memory state if the update has failed
	if err != nil {
		loadMeta(db, meta)
	}
	return err
}

func writePages(db *KV) error {
	// extend the mmap if needed
	size := (db.page.flushed + db.page.nappend) * BTREE_PAGE_SIZE
	if err := extendMmap(db, int(size)); err != nil {
		return err
	}
	// write data to the file
	for ptr, node := range db.page.updates {
		offset := int64(ptr * BTREE_PAGE_SIZE)
		if _, err := unix.Pwrite(db.fd, node, offset); err != nil {
			return err
		}
	}
	// discard in-memory data
	db.page.flushed += db.page.nappend
	db.page.nappend = 0
	db.page.updates = map[uint64][]byte{}
	return nil
}

// KV interfaces
func (db *KV) Get(key []byte) ([]byte, bool) {
	return db.tree.Get(key)
}
func (db *KV) Set(key []byte, val []byte) error {
	db.tree.Upsert(key, val)
	return updateFile(db)
}
func (db *KV) Update(key []byte, val []byte, mode int) (bool, error) {
	req := &InsertReq{Key: key, Val: val, Mode: mode}
	db.tree.Update(req)
	return req.Added, updateFile(db)
}
func (db *KV) Del(key []byte) (bool, error) {
	deleted := db.tree.Delete(key)
	return deleted, updateFile(db)
}

// cleanups
func (db *KV) Close() {
	for _, chunk := range db.mmap.chunks {
		err := syscall.Munmap(chunk)
		assert(err == nil)
	}
	_ = syscall.Close(db.fd)
}
