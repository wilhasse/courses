package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"
	"webserver/database"

	"github.com/dgrijalva/jwt-go"
	"github.com/joho/godotenv"
	"golang.org/x/crypto/bcrypt"
)

type apiConfig struct {
	fileserverHits int
}

type App struct {
	DB        *database.DB
	DBUser    *database.DBUser
	JwtSecret string
	PolkaKey  string
}

func main() {

	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Error loading .env file")
	}

	jwtSecret := os.Getenv("JWT_SECRET")
	polkaKey := os.Getenv("POLKA_KEY")
	log.Printf("Secredt: %s", jwtSecret)

	apiCfg := apiConfig{}
	db, _ := database.NewDB("database.json")
	dbUser, _ := database.NewUserDB("database_user.json")
	app := App{DB: db, DBUser: dbUser, JwtSecret: jwtSecret, PolkaKey: polkaKey}

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
	m.HandleFunc("POST /api/login", app.loginUsers)
	m.HandleFunc("PUT /api/users", app.updateUser)
	m.HandleFunc("POST /api/refresh", app.refreshUser)
	m.HandleFunc("POST /api/revoke", app.revokeUser)
	m.HandleFunc("DELETE /api/chirps/{chirps}", app.deleteChirpId)
	m.HandleFunc("/api/polka/webhooks", app.polkaWebhooks)

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
	err = srv.ListenAndServe()
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
	passwordHash, _ := bcrypt.GenerateFromPassword([]byte(params.Password), bcrypt.DefaultCost)
	respBody, _ := app.DBUser.CreateUser(params.Email, string(passwordHash))
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

func (app *App) updateUser(w http.ResponseWriter, r *http.Request) {
	tokenString := r.Header.Get("Authorization")
	if tokenString == "" {
		http.Error(w, "Missing token", http.StatusUnauthorized)
		return
	}

	tokenString = tokenString[len("Bearer "):]

	claims := &jwt.StandardClaims{}
	token, err := jwt.ParseWithClaims(tokenString, claims, func(token *jwt.Token) (interface{}, error) {
		return []byte(app.JwtSecret), nil
	})
	if err != nil || !token.Valid {
		http.Error(w, "Invalid token", http.StatusUnauthorized)
		return
	}

	// Check if the token is expired
	if claims.ExpiresAt < time.Now().Unix() {
		http.Error(w, "Token is expired", http.StatusUnauthorized)
		return
	}

	userID := []byte(claims.Subject)
	var requestBody struct {
		Email    string `json:"email"`
		Password string `json:"password"`
	}

	err = json.NewDecoder(r.Body).Decode(&requestBody)
	if err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	// Update the user in the database
	passwordHash, _ := bcrypt.GenerateFromPassword([]byte(requestBody.Password), bcrypt.DefaultCost)
	updatedUser, err := app.DBUser.UpdateUser(int(userID[0]), requestBody.Email, string(passwordHash))
	if err != nil {
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	responseBody := struct {
		ID    int    `json:"id"`
		Email string `json:"email"`
	}{
		ID:    updatedUser.ID,
		Email: updatedUser.Email,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(responseBody)
}

func (app *App) loginUsers(w http.ResponseWriter, r *http.Request) {

	decoder := json.NewDecoder(r.Body)
	params := database.User{}
	err := decoder.Decode(&params)
	if err != nil {
		log.Printf("Error decoding parameters: %s", err)
		http.Error(w, "Invalid request", http.StatusBadRequest) // More appropriate status code
		return
	}

	var code int
	respBody, _ := app.DBUser.GetUser(params.Email)
	respBody.ExpiresInSeconds = params.ExpiresInSeconds

	// check password
	err = bcrypt.CompareHashAndPassword([]byte(respBody.Password), []byte(params.Password))
	if err == nil {
		code = 200
	} else {
		code = 401
	}

	// Set expiration time
	var expirationTime time.Duration
	if respBody.ExpiresInSeconds > 0 {
		expirationTime = time.Duration(respBody.ExpiresInSeconds) * time.Second
		if expirationTime > 24*time.Hour {
			expirationTime = 24 * time.Hour
		}
	} else {
		expirationTime = 24 * time.Hour
	}

	claims := &jwt.StandardClaims{
		Issuer:    "chirpy",
		IssuedAt:  time.Now().UTC().Unix(),
		ExpiresAt: time.Now().Add(expirationTime).UTC().Unix(),
		Subject:   string(respBody.ID),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString([]byte(app.JwtSecret))
	if err != nil {
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Generate and store the refresh token
	refreshToken, err := generateRefreshToken()

	if err != nil {
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Store refresh token
	app.DBUser.StoreRefreshToken(respBody.ID, refreshToken)

	// Create a new struct without the password field
	userResp := database.UserResponse{
		Email:        respBody.Email,
		ID:           respBody.ID,
		Token:        tokenString,
		RefreshToken: refreshToken,
		IsChirpyRed:  respBody.IsChirpyRed,
	}

	dat, err := json.Marshal(userResp)
	if err != nil {
		log.Printf("Error marshalling JSON: %s", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(dat)
}

func generateRefreshToken() (string, error) {
	b := make([]byte, 32)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func (app *App) refreshUser(w http.ResponseWriter, r *http.Request) {

	tokenString := r.Header.Get("Authorization")
	if tokenString == "" {
		http.Error(w, "Missing token", http.StatusUnauthorized)
		return
	}

	tokenString = tokenString[len("Bearer "):]

	user, err := app.DBUser.GetUserByRefreshToken(tokenString)
	if err != nil {
		http.Error(w, "Invalid or expired refresh token", http.StatusUnauthorized)
		return
	}

	// Create a new access token
	expirationTime := time.Now().Add(1 * time.Hour)
	claims := &jwt.StandardClaims{
		Issuer:    "chirpy",
		IssuedAt:  time.Now().UTC().Unix(),
		ExpiresAt: expirationTime.UTC().Unix(),
		Subject:   string(user.ID),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err = token.SignedString([]byte(app.JwtSecret))
	if err != nil {
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	response := struct {
		Token string `json:"token"`
	}{
		Token: tokenString,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (app *App) revokeUser(w http.ResponseWriter, r *http.Request) {

	tokenString := r.Header.Get("Authorization")
	if tokenString == "" {
		http.Error(w, "Missing token", http.StatusUnauthorized)
		return
	}

	tokenString = tokenString[len("Bearer "):]

	err := app.DBUser.RevokeRefreshToken(tokenString)
	if err != nil {
		http.Error(w, "Invalid or non-existent refresh token", http.StatusUnauthorized)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (app *App) createChirps(w http.ResponseWriter, r *http.Request) {

	tokenString := r.Header.Get("Authorization")
	if tokenString == "" {
		http.Error(w, "Missing token", http.StatusUnauthorized)
		return
	}

	tokenString = tokenString[len("Bearer "):]

	claims := &jwt.StandardClaims{}
	token, err := jwt.ParseWithClaims(tokenString, claims, func(token *jwt.Token) (interface{}, error) {
		return []byte(app.JwtSecret), nil
	})
	if err != nil || !token.Valid {
		http.Error(w, "Invalid token", http.StatusUnauthorized)
		return
	}

	userID := int([]byte(claims.Subject)[0])

	decoder := json.NewDecoder(r.Body)
	params := database.Chirp{}
	err = decoder.Decode(&params)
	if err != nil {
		log.Printf("Error decoding parameters: %s", err)
		http.Error(w, "Invalid request", http.StatusBadRequest) // More appropriate status code
		return
	}

	var code int
	if len(params.Body) > 140 || userID == 0 {
		code = 400
	} else {
		respBody, _ := app.DB.CreateChirp(params.Body, userID)
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

	author_id, _ := strconv.Atoi(r.URL.Query().Get("author_id"))
	respBody, _ := app.DB.GetChirps(author_id)

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

func (app *App) deleteChirpId(w http.ResponseWriter, r *http.Request) {

	tokenString := r.Header.Get("Authorization")
	if tokenString == "" {
		http.Error(w, "Missing token", http.StatusUnauthorized)
		return
	}

	// token
	tokenString = tokenString[len("Bearer "):]

	claims := &jwt.StandardClaims{}
	token, err := jwt.ParseWithClaims(tokenString, claims, func(token *jwt.Token) (interface{}, error) {
		return []byte(app.JwtSecret), nil
	})
	if err != nil || !token.Valid {
		http.Error(w, "Invalid token", http.StatusUnauthorized)
		return
	}

	// user
	userID := int([]byte(claims.Subject)[0])

	// chirp id
	id, _ := strconv.Atoi(r.PathValue("chirps"))
	respBody, _ := app.DB.GetChirpId(id)

	if respBody.ID == 0 {

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(404)
		return
	}

	// verify ownership
	if userID != respBody.AuthorId {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(403)
		return
	}

	// remove
	err = app.DB.DeleteChirp(respBody.ID)
	if err != nil {
		http.Error(w, "Error", http.StatusInternalServerError)
		return
	}

	// body
	dat, err := json.Marshal(respBody)
	if err != nil {
		log.Printf("Error marshalling JSON: %s", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(204)
	w.Write(dat)
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

func (app *App) polkaWebhooks(w http.ResponseWriter, r *http.Request) {

	tokenString := r.Header.Get("Authorization")
	if tokenString == "" {
		http.Error(w, "Missing token", http.StatusUnauthorized)
		return
	}

	tokenString = tokenString[len("Bearer "):]

	// valid api ?
	if app.PolkaKey != tokenString {
		http.Error(w, "Invalid token", http.StatusUnauthorized)
		return
	}

	var requestBody struct {
		Event string `json:"event"`
		Data  struct {
			UserID int `json:"user_id"`
		} `json:"data"`
	}

	err := json.NewDecoder(r.Body).Decode(&requestBody)
	if err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	if requestBody.Event != "user.upgraded" {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	err = app.DBUser.UpdateUserMembership(requestBody.Data.UserID, true)
	if err != nil {
		if err.Error() == "user not found" {
			http.Error(w, "User not found", http.StatusNotFound)
		} else {
			http.Error(w, "Internal server error", http.StatusInternalServerError)
		}
		return
	}

	w.WriteHeader(http.StatusNoContent)
}
