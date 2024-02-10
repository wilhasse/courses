import { readable } from'svelte/store';

export default readable(new Date, callback => {

        callback(new Date);

        const interval = setInterval(() => {
            callback(new Date);
        }, 1000);

        return () => clearInterval(interval);

});