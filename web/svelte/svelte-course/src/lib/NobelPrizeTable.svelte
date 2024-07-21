<script>
    export let prizes = [];
</script>

<table>
    <thead><tr><th>year</th><th>category</th><th>laureates</th></tr></thead>
    <tbody>	
        {#each prizes as { year, category, laureates, overallMotivation }}
            <tr>
                <td>{year}</td>
                <td>{category}</td>
                <td>
                    {#if laureates}
                        <ul>
                            {#each laureates as { firstname, surname, motivation }}
                                <li>{@html firstname} {@html surname}: {motivation}</li>
                            {/each}
                        </ul>
                    {:else if overallMotivation}
                        {overallMotivation}
                    {:else}
                        N/A
                    {/if}
                </td>
            </tr>
        {/each}
    </tbody>
</table>