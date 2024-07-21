export function fetchPrizes(category) {
	if (category) {
		return fetchPrizesByCategory(category);
	} else {
		return fetchAllPrizes();
	}
}

export async function fetchAllPrizes() {
	const req = await fetch('https://api.nobelprize.org/v1/prize.json');
	return req.json();
}

export async function fetchPrizesByCategory(category) {
	const req = await fetch(`https://api.nobelprize.org/v1/prize.json?category=${category}`);
	return req.json();
}
