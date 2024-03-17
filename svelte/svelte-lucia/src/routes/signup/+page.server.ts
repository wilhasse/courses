import { lucia } from "$lib/server/auth";
import { fail, redirect } from "@sveltejs/kit";
import { generateId } from "lucia";
import { Argon2id } from "oslo/password";
import { db } from "$lib/server/db"; // Ensure this is your MySQL db connection

import type { Actions, PageServerLoad } from "./$types";

export const load: PageServerLoad = async (event) => {
	if (event.locals.user) {
		return redirect(302, "/");
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
			return fail(400, { message: 'Invalid username' });
		}

		if (typeof password !== 'string' || password.length < 6 || password.length > 255) {
			return fail(400, { message: 'Invalid password' });
		}

		const hashedPassword = await new Argon2id().hash(password);
		const userId = generateId(15);

		try {
			// Here you should use your MySQL query execution method
			// This is a generic example, adjust based on your db utility or ORM
			await db.execute('INSERT INTO user (id, username, password) VALUES (?, ?, ?)', [
				userId,
				username,
				hashedPassword
			]);

			const session = await lucia.createSession(userId, {});
			const sessionCookie = lucia.createSessionCookie(session.id);
			event.cookies.set(sessionCookie.name, sessionCookie.value, {
				path: '.',
				...sessionCookie.attributes
			});
		} catch (e: unknown) {
			// Adjust error handling for MySQL
			if (e instanceof Error && 'code' in e && e.code === 'ER_DUP_ENTRY') {
				return fail(400, { message: 'Username already used' });
			}
			console.error(e); // Ensure you log or handle the error appropriately
			return fail(500, { message: 'An unknown error occurred' });
		}
		return redirect(302, '/');
	}
};
