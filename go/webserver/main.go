package main

import (
	"fmt"
	"log"
	"net/http"
	"time"
)

func main() {

	m := http.NewServeMux()
	m.Handle("/app/", http.StripPrefix("/app/", http.FileServer(http.Dir("."))))
	m.Handle("/", http.FileServer(http.Dir(".")))
	m.HandleFunc("/healthz", handleOk)

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

func handleOk(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(200)
	const page = "OK"
	w.Write([]byte(page))
}
