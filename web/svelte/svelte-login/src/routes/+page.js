export const load = async ({ parent }) => {
    const { welcome_message } = await parent();
    return {
        message: welcome_message
    };
};