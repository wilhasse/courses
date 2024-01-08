<template>
    <h1>Welcome to the Forum</h1>
    <CategoryList :categories="categories"/>
</template>

<script>

import CategoryList from '@/components/CategoryList.vue'
export default {
  components: {
    CategoryList
  },
  computed: {
    categories () {
      return this.$store.state.categories
    }
  }
}
</script>

<style scoped>

</style>