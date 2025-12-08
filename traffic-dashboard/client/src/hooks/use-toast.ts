export const useToast = () => {
  return {
    toast: ({ title, description }: { title: string; description: string }) => {
      console.log(`${title}: ${description}`);
    }
  };
};