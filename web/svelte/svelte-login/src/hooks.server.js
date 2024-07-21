import { redirect } from '@sveltejs/kit';

const unProtectedRoutes = ['/','/login'];

export const handle = async ({ event, resolve }) => {
	const user = event.cookies.get('username');
	if (!user && !unProtectedRoutes.includes(event.url.pathname)) {
		throw redirect(303, '/');
	}
	if (user) {
        console.log("Username: " + user);
        event.locals.user = {
			user: user
		};        
	} else {
        console.log("I don't have user");
		if (!unProtectedRoutes.includes(event.url.pathname)) {
			throw redirect(303, '/');
		}
	}

    const query = event.url.searchParams.get('signout');
	if (Boolean(query) == true) {
        console.log("Removed cookie username");
		await event.cookies.delete('username', { path: '/' });
	}

	return resolve(event);
};