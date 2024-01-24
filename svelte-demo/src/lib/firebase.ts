// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCcaXgXJJXsXS1bFhfl0FjsKh0uevNwGjY",
  authDomain: "svelte-demo-4e71d.firebaseapp.com",
  databaseURL: "https://svelte-demo-4e71d-default-rtdb.firebaseio.com",
  projectId: "svelte-demo-4e71d",
  storageBucket: "svelte-demo-4e71d.appspot.com",
  messagingSenderId: "746206817349",
  appId: "1:746206817349:web:b96716f2c859a5f9a7c72c",
  measurementId: "G-NCHC84BGJV"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
