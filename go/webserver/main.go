package main

import (
	"fmt"
	"log"
	"net/http"
	"time"
)

func main() {

	m := http.NewServeMux()
	m.Handle("/", http.FileServer(http.Dir(".")))

	const addr = ":8080"
	srv := http.Server{
		Handler:      m,
		Addr:         addr,
		WriteTimeout: 30 * time.Second,
		ReadTimeout:  30 * time.Second,
	}

	// this blocks forever, until the server
	// has an unrecoverable error
	fmt.Println("server started on ", addr)
	err := srv.ListenAndServe()
	log.Fatal(err)
}
