import { readable } from 'svelte/store';

export default readable(new Date, set => {
	const interval = setInterval(() => {
		set(new Date);
	}, 1000);

	return () => clearInterval(interval);
});