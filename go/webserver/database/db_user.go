package database

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"os"
	"sync"
)

type User struct {
	ID               int    `json:"id"`
	Email            string `json:"email"`
	Password         string `json:"password"`
	ExpiresInSeconds int64  `json:"expires_in_seconds"`
	RefreshToken     string `json:"refresh_token"`
}

type UserResponse struct {
	ID           int    `json:"id"`
	Email        string `json:"email"`
	Token        string `json:"token"`
	RefreshToken string `json:"refresh_token"`
}

type DBUser struct {
	Path string
	Mux  *sync.RWMutex
}

type DBUserStructure struct {
	Users map[int]User `json:"users"`
}

func NewUserDB(path string) (*DBUser, error) {
	db := &DBUser{
		Path: path,
		Mux:  new(sync.RWMutex),
	}
	err := db.ensureDB()
	if err != nil {
		return nil, err
	}
	return db, nil
}

func (db *DBUser) CreateUser(email string, password string) (User, error) {
	db.Mux.Lock()
	defer db.Mux.Unlock()

	dbData, err := db.loadDB()
	if err != nil {
		return User{}, err
	}

	newID := len(dbData.Users) + 1
	user := User{
		ID:       newID,
		Email:    email,
		Password: password,
	}

	dbData.Users[newID] = user
	err = db.writeDB(dbData)
	if err != nil {
		return User{}, err
	}

	return user, nil
}

func (db *DBUser) UpdateUser(id int, email string, password string) (User, error) {
	db.Mux.Lock()
	defer db.Mux.Unlock()

	dbData, err := db.loadDB()
	if err != nil {
		return User{}, err
	}

	user, exists := dbData.Users[id]
	if !exists {
		return User{}, errors.New("user not found")
	}

	user.Email = email
	user.Password = password
	dbData.Users[id] = user

	err = db.writeDB(dbData)
	if err != nil {
		return User{}, err
	}

	return user, nil
}

func (db *DBUser) StoreRefreshToken(userID int, refreshToken string) error {

	db.Mux.Lock()
	defer db.Mux.Unlock()

	dbData, err := db.loadDB()
	if err != nil {
		return err
	}

	user, exists := dbData.Users[userID]
	if !exists {
		return errors.New("user not found")
	}

	user.RefreshToken = refreshToken
	dbData.Users[userID] = user

	return db.writeDB(dbData)
}

func (db *DBUser) GetUserByRefreshToken(refreshToken string) (User, error) {
	db.Mux.RLock()
	defer db.Mux.RUnlock()

	dbData, err := db.loadDB()
	if err != nil {
		return User{}, err
	}

	for _, user := range dbData.Users {
		if user.RefreshToken == refreshToken {
			return user, nil
		}
	}

	return User{}, errors.New("refresh token not found")
}

func (db *DBUser) RevokeRefreshToken(refreshToken string) error {
	db.Mux.Lock()
	defer db.Mux.Unlock()

	dbData, err := db.loadDB()
	if err != nil {
		return err
	}

	for id, user := range dbData.Users {
		if user.RefreshToken == refreshToken {
			user.RefreshToken = ""
			dbData.Users[id] = user
			return db.writeDB(dbData)
		}
	}

	return errors.New("refresh token not found")
}

func (db *DBUser) GetUser(Email string) (User, error) {
	db.Mux.RLock()
	defer db.Mux.RUnlock()

	dbData, err := db.loadDB()
	if err != nil {
		return User{}, err
	}

	for _, user := range dbData.Users {
		if user.Email == Email {
			return user, nil
		}
	}

	return User{}, nil
}

func (db *DBUser) ensureDB() error {
	if _, err := os.Stat(db.Path); errors.Is(err, os.ErrNotExist) {
		return db.writeDB(DBUserStructure{Users: make(map[int]User)})
	}
	return nil
}

func (db *DBUser) loadDB() (DBUserStructure, error) {
	file, err := ioutil.ReadFile(db.Path)
	if err != nil {
		return DBUserStructure{}, err
	}

	var dbData DBUserStructure
	err = json.Unmarshal(file, &dbData)
	if err != nil {
		return DBUserStructure{}, err
	}

	return dbData, nil
}

func (db *DBUser) writeDB(DBUserStructure DBUserStructure) error {
	data, err := json.Marshal(DBUserStructure)
	if err != nil {
		return err
	}

	return ioutil.WriteFile(db.Path, data, 0644)
}
