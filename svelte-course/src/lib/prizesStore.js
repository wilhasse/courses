import { writable } from 'svelte/store';
import { fetchAllPrizes } from './prizes.js';

const { set, update, subscribe } = writable(null);

fetchAllPrizes().then(
	({ prizes }) => set(prizes),
	error => set({ error })
);

export default {
	subscribe,

    addPrize(prize) {
        update($prizes => {
            return [prize, ...$prizes];
        });
    }
};

