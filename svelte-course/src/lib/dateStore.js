export default {
    subscribe(callback) {
        callback(new Date);

        const interval = setInterval(() => {
            callback(new Date);
        }, 1000);

        return () => clearInterval(interval);
    }
};