// @ts-ignore
export function shuffle(array) {
	let i = array.length;

	while (i--) {
		const j = Math.floor(Math.random() * i + 1);
		const temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}

	return array;
}