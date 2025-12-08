// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCahTAAw0-CZus_-b0LqE9K_t6n3TWV350",
  authDomain: "traffic-analyser-fad30.firebaseapp.com",
  databaseURL: "https://traffic-analyser-fad30-default-rtdb.firebaseio.com",
  projectId: "traffic-analyser-fad30",
  storageBucket: "traffic-analyser-fad30.firebasestorage.app",
  messagingSenderId: "442603118634",
  appId: "1:442603118634:web:cf71a305e4fe4ca0a8a353"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

export default db;