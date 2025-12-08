export const useToast = () => {
  const toast = (notification: { title: string; description: string }) => {
    console.log(`Toast: ${notification.title} - ${notification.description}`);
    // You can add a real toast notification library here later (like react-toastify, sonner, etc.)
  };

  return { toast };
};