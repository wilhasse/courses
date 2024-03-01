//src/routes/+layout.server.js
export const load = async ({locals}) => {
    return {
        user: locals.user,
        welcome_message: "welcome back",
    };
};