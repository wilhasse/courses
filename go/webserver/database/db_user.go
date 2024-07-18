package database

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"os"
	"sync"
)

type User struct {
	ID    int    `json:"id"`
	Email string `json:"email"`
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

func (db *DBUser) CreateUser(email string) (User, error) {
	db.Mux.Lock()
	defer db.Mux.Unlock()

	dbData, err := db.loadDB()
	if err != nil {
		return User{}, err
	}

	newID := len(dbData.Users) + 1
	user := User{
		ID:    newID,
		Email: email,
	}

	dbData.Users[newID] = user
	err = db.writeDB(dbData)
	if err != nil {
		return User{}, err
	}

	return user, nil
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
