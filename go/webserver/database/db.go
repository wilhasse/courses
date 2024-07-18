package database

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"os"
	"sync"
)

type Chirp struct {
	ID   int    `json:"id"`
	Body string `json:"body"`
}

type DB struct {
	Path string
	Mux  *sync.RWMutex
}

type DBStructure struct {
	Chirps map[int]Chirp `json:"chirps"`
}

func NewDB(path string) (*DB, error) {
	db := &DB{
		Path: path,
		Mux:  new(sync.RWMutex),
	}
	err := db.ensureDB()
	if err != nil {
		return nil, err
	}
	return db, nil
}

func (db *DB) CreateChirp(body string) (Chirp, error) {
	db.Mux.Lock()
	defer db.Mux.Unlock()

	dbData, err := db.loadDB()
	if err != nil {
		return Chirp{}, err
	}

	newID := len(dbData.Chirps) + 1
	chirp := Chirp{
		ID:   newID,
		Body: body,
	}

	dbData.Chirps[newID] = chirp
	err = db.writeDB(dbData)
	if err != nil {
		return Chirp{}, err
	}

	return chirp, nil
}

func (db *DB) GetChirps() ([]Chirp, error) {
	db.Mux.RLock()
	defer db.Mux.RUnlock()

	dbData, err := db.loadDB()
	if err != nil {
		return nil, err
	}

	chirps := make([]Chirp, 0, len(dbData.Chirps))
	for _, chirp := range dbData.Chirps {
		chirps = append(chirps, chirp)
	}

	return chirps, nil
}

func (db *DB) ensureDB() error {
	if _, err := os.Stat(db.Path); errors.Is(err, os.ErrNotExist) {
		return db.writeDB(DBStructure{Chirps: make(map[int]Chirp)})
	}
	return nil
}

func (db *DB) loadDB() (DBStructure, error) {
	file, err := ioutil.ReadFile(db.Path)
	if err != nil {
		return DBStructure{}, err
	}

	var dbData DBStructure
	err = json.Unmarshal(file, &dbData)
	if err != nil {
		return DBStructure{}, err
	}

	return dbData, nil
}

func (db *DB) writeDB(dbStructure DBStructure) error {
	data, err := json.Marshal(dbStructure)
	if err != nil {
		return err
	}

	return ioutil.WriteFile(db.Path, data, 0644)
}
