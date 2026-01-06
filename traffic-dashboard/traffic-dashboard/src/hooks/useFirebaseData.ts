import { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { setTrafficData } from '../store';
// @ts-ignore - JavaScript module without type declarations
import { subscribeToProcessedData } from '../firebase-config';

/**
 * Custom hook to subscribe to Firebase Realtime Database
 * and automatically update Redux store with processed traffic data
 */
export const useFirebaseData = () => {
    const dispatch = useDispatch();

    useEffect(() => {
        console.log('[Firebase] Subscribing to processed_data from backend...');

        // Subscribe to Firebase realtime updates
        const unsubscribe = subscribeToProcessedData((data: any) => {
            console.log('[Firebase] Received data:', data);
            dispatch(setTrafficData(data));
        });

        // Cleanup subscription on unmount
        return () => {
            console.log('[Firebase] Unsubscribing...');
            unsubscribe();
        };
    }, [dispatch]);
};

export default useFirebaseData;
