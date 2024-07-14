package main

import (
	"fmt"
	"log"
	"net/http"
	"strconv"
	"time"
)

type apiConfig struct {
	fileserverHits int
}

func main() {

	apiCfg := apiConfig{}

	m := http.NewServeMux()
	m.Handle("/app", apiCfg.middlewareMetricsInc(http.StripPrefix("/app/", http.FileServer(http.Dir(".")))))
	m.Handle("/", apiCfg.middlewareMetricsInc(http.FileServer(http.Dir("."))))
	m.HandleFunc("/healthz", handleOk)
	m.HandleFunc("/reset", apiCfg.reset)
	m.HandleFunc("/metrics", apiCfg.metrics)

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

func (cfg *apiConfig) middlewareMetricsInc(next http.Handler) http.Handler {
	cfg.fileserverHits += 1
	return next
}

func (cfg *apiConfig) reset(w http.ResponseWriter, r *http.Request) {
	cfg.fileserverHits = 0
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(200)
}

func (cfg *apiConfig) metrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(200)
	page := "Hits:" + strconv.Itoa(cfg.fileserverHits)
	w.Write([]byte(page))
}

func handleOk(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(200)
	const page = "OK"
	w.Write([]byte(page))
}
