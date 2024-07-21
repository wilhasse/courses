<script>
    export let data;
</script>

<h1>Hi, {data.message}</h1>

<!-- Using an anchor tag styled as a button -->
<a href="/login" class="button">Please log-in</a>

<style>
    .button {
        display: inline-block;
        padding: 10px 20px;
        margin: 10px 0;
        border: none;
        border-radius: 5px;
        background-color: #007bff;
        color: white;
        text-align: center;
        text-decoration: none;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }

    .button:hover {
        background-color: #0056b3;
    }
</style>
