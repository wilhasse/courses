// +build ignore

package main

import (
	"encoding/binary"
	"bytes"
	"fmt"
)

// Constants from the B-tree implementation
const HEADER = 4
const BTREE_PAGE_SIZE = 256
const BTREE_MAX_KEY_SIZE = 50
const BTREE_MAX_VAL_SIZE = 150

const (
	BNODE_NODE = 1 // internal nodes without values
	BNODE_LEAF = 2 // leaf nodes with values
)

type BNode []byte

type BTree struct {
	root uint64
	get func(uint64) []byte
	new func([]byte) uint64
	del func(uint64)
}

// Include all the necessary B-tree methods
func (node BNode) btype() uint16 {
	return binary.LittleEndian.Uint16(node[0:2])
}

func (node BNode) nkeys() uint16 {
	return binary.LittleEndian.Uint16(node[2:4])
}

func (node BNode) setHeader(btype uint16, nkeys uint16) {
	binary.LittleEndian.PutUint16(node[0:2], btype)
	binary.LittleEndian.PutUint16(node[2:4], nkeys)
}

func (node BNode) getPtr(idx uint16) uint64 {
	assert(idx < node.nkeys())
	pos := HEADER + 8*idx
	return binary.LittleEndian.Uint64(node[pos:])
}

func (node BNode) setPtr(idx uint16, val uint64) {
	assert(idx < node.nkeys())
	pos := HEADER + 8*idx
	binary.LittleEndian.PutUint64(node[pos:], val)
}

func offsetPos(node BNode, idx uint16) uint16 {
	assert(1 <= idx && idx <= node.nkeys())
	return HEADER + 8*node.nkeys() + 2*(idx-1)
}

func (node BNode) getOffset(idx uint16) uint16 {
	if idx == 0 {
		return 0
	}
	return binary.LittleEndian.Uint16(node[offsetPos(node, idx):])
}

func (node BNode) setOffset(idx uint16, offset uint16) {
	binary.LittleEndian.PutUint16(node[offsetPos(node, idx):], offset)
}

func (node BNode) kvPos(idx uint16) uint16 {
	assert(idx <= node.nkeys())
	base := HEADER + 8*node.nkeys() + 2*node.nkeys()
	return base + node.getOffset(idx)
}

func (node BNode) getKey(idx uint16) []byte {
	assert(idx < node.nkeys())
	pos := node.kvPos(idx)
	klen := binary.LittleEndian.Uint16(node[pos:])
	if node.btype() == BNODE_LEAF {
		return node[pos+4:][:klen]
	} else {
		return node[pos+2:][:klen]
	}
}

func (node BNode) getVal(idx uint16) []byte {
	if node.btype() == BNODE_LEAF {
		pos := node.kvPos(idx)
		klen := binary.LittleEndian.Uint16(node[pos:])
		vlen := binary.LittleEndian.Uint16(node[pos+2:])
		return node[pos+4+klen:][:vlen]
	} else {
		return nil
	}
}

func (node BNode) nbytes() uint16 {
	return node.kvPos(node.nkeys())
}

func assert(cond bool) {
	if !cond {
		panic("assertion failure")
	}
}

func nodeLookupLE(node BNode, key []byte) uint16 {
	nkeys := node.nkeys()
	found := uint16(0)
	for i := uint16(1); i < nkeys; i++ {
		cmp := bytes.Compare(node.getKey(i), key)
		if cmp <= 0 {
			found = i
		}
		if cmp >= 0 {
			break
		}
	}
	return found
}

func nodeAppendKV(new BNode, idx uint16, ptr uint64, key []byte, val []byte, btype uint16) {
	new.setPtr(idx, ptr)
	pos := new.kvPos(idx)
	binary.LittleEndian.PutUint16(new[pos+0:], uint16(len(key)))
	if btype == BNODE_LEAF {
		binary.LittleEndian.PutUint16(new[pos+2:], uint16(len(val)))
		copy(new[pos+4:], key)
		copy(new[pos+4+uint16(len(key)):], val)
		new.setOffset(idx+1, new.getOffset(idx)+4+uint16(len(key)+len(val)))
	} else {
		copy(new[pos+2:], key)
		new.setOffset(idx+1, new.getOffset(idx)+2+uint16(len(key)))
	}
}

func nodeAppendRange(new BNode, old BNode, dstNew uint16, srcOld uint16, n uint16, btype uint16) {
	assert(srcOld+n <= old.nkeys())
	assert(dstNew+n <= new.nkeys())
	if n == 0 {
		return
	}
	for i := uint16(0); i < n; i++ {
		new.setPtr(dstNew+i, old.getPtr(srcOld+i))
	}
	dstBegin := new.getOffset(dstNew)
	srcBegin := old.getOffset(srcOld)
	for i := uint16(1); i <= n; i++ {
		offset := dstBegin + old.getOffset(srcOld+i) - srcBegin
		new.setOffset(dstNew+i, offset)
	}
	begin := old.kvPos(srcOld)
	end := old.kvPos(srcOld + n)
	copy(new[new.kvPos(dstNew):], old[begin:end])
}

func leafInsert(new BNode, old BNode, idx uint16, key []byte, val []byte) {
	new.setHeader(BNODE_LEAF, old.nkeys()+1)
	nodeAppendRange(new, old, 0, 0, idx, BNODE_LEAF)
	nodeAppendKV(new, idx, 0, key, val, BNODE_LEAF)
	nodeAppendRange(new, old, idx+1, idx, old.nkeys()-idx, BNODE_LEAF)
}

func leafUpdate(new BNode, old BNode, idx uint16, key []byte, val []byte) {
	new.setHeader(BNODE_LEAF, old.nkeys())
	nodeAppendRange(new, old, 0, 0, idx, BNODE_LEAF)
	nodeAppendKV(new, idx, 0, key, val, BNODE_LEAF)
	nodeAppendRange(new, old, idx+1, idx+1, old.nkeys()-(idx+1), BNODE_LEAF)
}

func leafDelete(new BNode, old BNode, idx uint16) {
	new.setHeader(BNODE_LEAF, old.nkeys()-1)
	nodeAppendRange(new, old, 0, 0, idx, BNODE_LEAF)
	nodeAppendRange(new, old, idx, idx+1, old.nkeys()-(idx+1), BNODE_LEAF)
}

// Simplified insert for demo
func (tree *BTree) Insert(key []byte, val []byte) {
	assert(len(key) != 0)
	assert(len(key) <= BTREE_MAX_KEY_SIZE)
	assert(len(val) <= BTREE_MAX_VAL_SIZE)

	if tree.root == 0 {
		root := BNode(make([]byte, BTREE_PAGE_SIZE))
		root.setHeader(BNODE_LEAF, 2)
		nodeAppendKV(root, 0, 0, nil, nil, BNODE_LEAF)
		nodeAppendKV(root, 1, 0, key, val, BNODE_LEAF)
		tree.root = tree.new(root)
		return
	}

	// For demo, we'll do simple leaf insertion without splitting
	node := tree.get(tree.root)
	new := BNode(make([]byte, BTREE_PAGE_SIZE))
	
	idx := nodeLookupLE(BNode(node), key)
	if bytes.Equal(key, BNode(node).getKey(idx)) {
		leafUpdate(new, BNode(node), idx, key, val)
	} else {
		leafInsert(new, BNode(node), idx+1, key, val)
	}
	
	tree.del(tree.root)
	tree.root = tree.new(new)
}

// Simplified delete for demo
func (tree *BTree) Delete(key []byte) bool {
	if tree.root == 0 {
		return false
	}
	
	node := tree.get(tree.root)
	idx := nodeLookupLE(BNode(node), key)
	
	if !bytes.Equal(key, BNode(node).getKey(idx)) {
		return false
	}
	
	new := BNode(make([]byte, BTREE_PAGE_SIZE))
	leafDelete(new, BNode(node), idx)
	
	tree.del(tree.root)
	tree.root = tree.new(new)
	return true
}

// Demo functions
func printTree(tree *BTree) {
	if tree.root == 0 {
		fmt.Println("  (empty tree)")
		return
	}
	
	node := BNode(tree.get(tree.root))
	nkeys := node.nkeys()
	
	for i := uint16(0); i < nkeys; i++ {
		key := node.getKey(i)
		val := node.getVal(i)
		if len(key) == 0 {
			fmt.Println("  [dummy key]")
		} else {
			fmt.Printf("  %s -> %s\n", string(key), string(val))
		}
	}
}

func main() {
	fmt.Println("B-tree Database Module 04 Demo")
	fmt.Println("==============================")
	
	// Create in-memory pages storage
	pages := make(map[uint64][]byte)
	nextPage := uint64(1)
	
	// Create B-tree instance
	tree := &BTree{
		get: func(ptr uint64) []byte {
			page, ok := pages[ptr]
			if !ok {
				panic(fmt.Sprintf("page not found: %d", ptr))
			}
			return page
		},
		new: func(node []byte) uint64 {
			ptr := nextPage
			nextPage++
			pages[ptr] = make([]byte, BTREE_PAGE_SIZE)
			copy(pages[ptr], node)
			return ptr
		},
		del: func(ptr uint64) {
			delete(pages, ptr)
		},
	}
	
	// Demo operations
	fmt.Println("\n1. Starting with empty tree")
	printTree(tree)
	
	fmt.Println("\n2. Inserting items:")
	items := []struct{ key, val string }{
		{"apple", "fruit"},
		{"banana", "fruit"},
		{"carrot", "vegetable"},
		{"date", "fruit"},
		{"eggplant", "vegetable"},
	}
	
	for _, item := range items {
		fmt.Printf("   Inserting: %s -> %s\n", item.key, item.val)
		tree.Insert([]byte(item.key), []byte(item.val))
	}
	
	fmt.Println("\n3. Current tree contents:")
	printTree(tree)
	
	fmt.Println("\n4. Updating an item:")
	fmt.Println("   Updating: apple -> red fruit")
	tree.Insert([]byte("apple"), []byte("red fruit"))
	
	fmt.Println("\n5. Tree after update:")
	printTree(tree)
	
	fmt.Println("\n6. Deleting items:")
	deleteKeys := []string{"banana", "date"}
	for _, key := range deleteKeys {
		fmt.Printf("   Deleting: %s\n", key)
		if !tree.Delete([]byte(key)) {
			fmt.Printf("   Failed to delete: %s\n", key)
		}
	}
	
	fmt.Println("\n7. Final tree contents:")
	printTree(tree)
	
	fmt.Printf("\nDemo complete! Tree has %d pages in memory.\n", len(pages))
}