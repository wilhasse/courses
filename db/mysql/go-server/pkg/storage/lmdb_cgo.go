package storage

/*
#cgo CFLAGS: -I${SRCDIR}/../../lmdb-lib/include
#cgo LDFLAGS: -L${SRCDIR}/../../lmdb-lib/lib -llmdb
#cgo linux LDFLAGS: -ldl
#cgo darwin LDFLAGS: -ldl
*/
import "C"