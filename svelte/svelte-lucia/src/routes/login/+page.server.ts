import { lucia } from '$lib/server/auth';
import { fail, redirect } from '@sveltejs/kit';
import { Argon2id } from 'oslo/password';
import { db } from '$lib/server/db';

import type { Actions, PageServerLoad } from './$types';
import type { DatabaseUser } from '$lib/server/db';

export const load: PageServerLoad = async (event) => {
	if (event.locals.user) {
		return redirect(302, '/');
	}
	return {};
};

export const actions: Actions = {
	default: async (event) => {
		const formData = await event.request.formData();
		const username = formData.get('username');
		const password = formData.get('password');

		if (
			typeof username !== 'string' ||
			username.length < 3 ||
			username.length > 31 ||
			!/^[a-z0-9_-]+$/.test(username)
		) {
			return fail(400, {
				message: 'Invalid username'
			});
		}
		if (typeof password !== 'string' || password.length < 6 || password.length > 255) {
			return fail(400, {
				message: 'Invalid password'
			});
		}

		// Use `.execute()` for prepared statements with mysql2/promise
		const [rows] = await db.execute('SELECT * FROM user WHERE username = ?', [username]);

		// `rows` will contain the result set. Check if a user was found.
		const existingUser: DatabaseUser | undefined = rows[0] ? (rows[0] as DatabaseUser) : undefined;
        console.log(existingUser);

		if (!existingUser) {
			return fail(400, {
				message: 'User does not exist'
			});
		}

		const validPassword = await new Argon2id().verify(existingUser.password, password);
		if (!validPassword) {
			return fail(400, {
				message: 'Incorrect Password'
			});
		}

		const session = await lucia.createSession(existingUser.id, {});
		const sessionCookie = lucia.createSessionCookie(session.id);
		event.cookies.set(sessionCookie.name, sessionCookie.value, {
			path: '.',
			...sessionCookie.attributes
		});

        console.log(session);
        console.log(sessionCookie);

		return redirect(302, '/');
	}
};
