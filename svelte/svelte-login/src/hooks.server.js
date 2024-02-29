import { redirect } from '@sveltejs/kit';

const unProtectedRoutes = ['/','/login'];

export const handle = async ({ event, resolve }) => {
	const sessionId = event.cookies.get('session_id');
	if (!sessionId && !unProtectedRoutes.includes(event.url.pathname)) {
		throw redirect(303, '/');
	}
	if (sessionId) {
        console.log("Session ID: " + sessionId);
	} else {
        console.log("I don't have sessionId");
		if (!unProtectedRoutes.includes(event.url.pathname)) {
			throw redirect(303, '/');
		}
	}

	return resolve(event);
};