import { redirect } from '@sveltejs/kit';
import { dev } from '$app/environment';

export const actions = {
	default: async ({ request, cookies }) => {
		const form = await request.formData();
		const user = form.get('user');
		const password = form.get('password');

        console.log(user, password);
		if (user === 'wil' && password === 'wil') {

            cookies.set('session_id', user, {
                path: '/',
                httpOnly: true,
                sameSite: 'strict',
                secure: !dev,
                maxAge: 60 * 60 * 24 * 7 // one week
            });

            throw redirect(307, '/');
        } 

        // login not ok
        throw redirect(303, '/'); 
	}
};