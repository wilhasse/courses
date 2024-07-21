import { derived } from 'svelte/store';
import prizes from './prizesStore.js';
import category from './categoryStore.js';

export default derived([category, prizes], ([$category, $prizes]) => {
	if ($prizes === null || $prizes.error) {
		return $prizes;
	}
	
	return $prizes.filter(prize => prize.category === $category);
});
