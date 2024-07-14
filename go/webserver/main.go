package main

import (
	"encoding/json"
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
	m.Handle("/app/*", http.StripPrefix("/app/", apiCfg.middlewareMetricsInc(http.FileServer(http.Dir(".")))))
	m.Handle("/", apiCfg.middlewareMetricsInc(http.FileServer(http.Dir("."))))
	m.HandleFunc("GET /api/healthz", handleOk)
	m.HandleFunc("GET /api/reset", apiCfg.reset)
	m.HandleFunc("GET /api/metrics", apiCfg.metrics)
	m.HandleFunc("GET /admin/metrics", apiCfg.adminMetrics)
	m.HandleFunc("POST /api/validate_chirp", validate)

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
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

		// Increment the counter on each request
		cfg.fileserverHits++
		// Continue to the next handler
		next.ServeHTTP(w, r)
	})
}

func (cfg *apiConfig) reset(w http.ResponseWriter, r *http.Request) {
	cfg.fileserverHits = 0
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(200)
}

func (cfg *apiConfig) metrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed) // 405
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(200)
	page := "Hits: " + strconv.Itoa(cfg.fileserverHits)
	w.Write([]byte(page))
}

func (cfg *apiConfig) adminMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed) // 405
		return
	}
	w.Header().Set("Content-Type", "text/html")
	w.WriteHeader(200)
	page := fmt.Sprintf(`<html>
<body>
    <h1>Welcome, Chirpy Admin</h1>
    <p>Chirpy has been visited %d times!</p>
</body>
</html>`, cfg.fileserverHits)
	w.Write([]byte(page))
}

func handleOk(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed) // 405
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(200)
	const page = "OK"
	w.Write([]byte(page))
}

// Define returnVals only once outside of the function for simplicity.
type returnVals struct {
	Error string `json:"error,omitempty"` // omitempty will omit this field if it's an empty string.
	Valid *bool  `json:"valid,omitempty"` // Use a pointer to bool to differentiate between omitted and false values.
}

func validate(w http.ResponseWriter, r *http.Request) {

	type parameters struct {
		Body string `json:"body"`
	}

	decoder := json.NewDecoder(r.Body)
	params := parameters{}
	err := decoder.Decode(&params)
	if err != nil {
		log.Printf("Error decoding parameters: %s", err)
		http.Error(w, "Invalid request", http.StatusBadRequest) // More appropriate status code
		return
	}

	respBody := returnVals{}
	var code int
	if len(params.Body) > 140 {
		respBody.Error = "Chirp is too long"
		code = 400
	} else {
		valid := true // Declare a true boolean value to use the address
		respBody.Valid = &valid
		code = 200
	}

	dat, err := json.Marshal(respBody)
	if err != nil {
		log.Printf("Error marshalling JSON: %s", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(dat)
}
