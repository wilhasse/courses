package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
	"webserver/database"
)

type apiConfig struct {
	fileserverHits int
}

type App struct {
	DB     *database.DB
	DBUser *database.DBUser
}

func main() {

	apiCfg := apiConfig{}
	db, _ := database.NewDB("database.json")
	dbUser, _ := database.NewUserDB("database_user.json")
	app := App{DB: db, DBUser: dbUser}

	m := http.NewServeMux()
	m.Handle("/app/*", http.StripPrefix("/app/", apiCfg.middlewareMetricsInc(http.FileServer(http.Dir(".")))))
	m.Handle("/", apiCfg.middlewareMetricsInc(http.FileServer(http.Dir("."))))
	m.HandleFunc("GET /api/healthz", handleOk)
	m.HandleFunc("GET /api/reset", apiCfg.reset)
	m.HandleFunc("GET /api/metrics", apiCfg.metrics)
	m.HandleFunc("GET /admin/metrics", apiCfg.adminMetrics)
	m.HandleFunc("GET /api/chirps", app.getChirps)
	m.HandleFunc("POST /api/chirps", app.createChirps)
	m.HandleFunc("GET /api/chirps/{chirps}", app.getChirpId)
	m.HandleFunc("POST /api/users", app.createUsers)

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

func (app *App) createUsers(w http.ResponseWriter, r *http.Request) {

	decoder := json.NewDecoder(r.Body)
	params := database.User{}
	err := decoder.Decode(&params)
	if err != nil {
		log.Printf("Error decoding parameters: %s", err)
		http.Error(w, "Invalid request", http.StatusBadRequest) // More appropriate status code
		return
	}

	var code int
	if len(params.Email) > 140 {
		code = 400
	} else {
		respBody, _ := app.DBUser.CreateUser(params.Email)
		code = 201

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
}

func (app *App) createChirps(w http.ResponseWriter, r *http.Request) {

	decoder := json.NewDecoder(r.Body)
	params := database.Chirp{}
	err := decoder.Decode(&params)
	if err != nil {
		log.Printf("Error decoding parameters: %s", err)
		http.Error(w, "Invalid request", http.StatusBadRequest) // More appropriate status code
		return
	}

	var code int
	if len(params.Body) > 140 {
		code = 400
	} else {
		respBody, _ := app.DB.CreateChirp(params.Body)
		code = 201

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
}

func (app *App) getChirps(w http.ResponseWriter, r *http.Request) {

	respBody, _ := app.DB.GetChirps()

	dat, err := json.Marshal(respBody)
	if err != nil {
		log.Printf("Error marshalling JSON: %s", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(200)
	w.Write(dat)
}

func (app *App) getChirpId(w http.ResponseWriter, r *http.Request) {

	id, _ := strconv.Atoi(r.PathValue("chirps"))
	respBody, _ := app.DB.GetChirpId(id)

	if respBody.ID == 0 {

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(404)
		return
	}
	dat, err := json.Marshal(respBody)
	if err != nil {
		log.Printf("Error marshalling JSON: %s", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(200)
	w.Write(dat)
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
	// omitempty will omit this field if it's an empty string.
	Error string `json:"error,omitempty"`
	// use a pointer to bool to differentiate between omitted and false values.
	Valid *bool `json:"valid,omitempty"`
	// text converted
	CleanedBody string `json:"cleaned_body"`
}

func clearText(source string) string {

	var badwords = []string{"kerfuffle", "sharbert", "fornax"}
	var splitWords = strings.Split(source, " ")

	for i, sword := range splitWords {

		for _, word := range badwords {

			if strings.ToLower(sword) == word {

				splitWords[i] = "****"
			}
		}

	}

	return strings.Join(splitWords, " ")
}
