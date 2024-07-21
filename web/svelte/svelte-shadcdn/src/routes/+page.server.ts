export const actions = {
    checkboxMultiple: async ({ request }) => {
      // Your logic to handle form submission, such as validating the data
      const formData = await request.formData();
      // Imagine you perform some validation and possibly encounter some errors
      console.log(formData);
  
      // If everything is fine, you might redirect or return some success response
      if (true) {
        return {
          // Redirect or indicate success as needed
          redirect: '/success-page',
          // No need to return form data if redirecting or if no errors
        };
      } else {
        // If there are errors, return them in a structured form
        const errors = {/* your validation errors */};
        
        // Return form data with errors so that sveltekit-superforms can process it
        return {
          form: {
            data: formData, // or the processed data you wish to return
            errors: errors,
          }
        };
      }
    }
  };
  