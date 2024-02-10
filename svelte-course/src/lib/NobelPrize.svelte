<script>

async function fetchNobelPrizeData() {
    const req = await fetch("https://api.nobelprize.org/v1/prize.json");
    return req.json();
}

</script>

<div>
{#await fetchNobelPrizeData() }
    Loading ...
{:then {prizes} } 
    <h1>Nobel Prize Winners</h1>
    <slot {prizes}></slot>
{:catch err}
    Error: {err}
{/await}
</div>

