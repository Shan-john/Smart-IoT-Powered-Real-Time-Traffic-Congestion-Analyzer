import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";
import DashboardPage from "@/pages/Dashboard";
import AdminPage from "@/pages/AdminPage";
import { Provider } from "react-redux";
import { store } from "./store";
import { useFirebaseData } from "./hooks/useFirebaseData";


function Router() {

  return (
    <Switch>
      <Route path="/" component={DashboardPage} />
      <Route path="/admin" component={AdminPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

// Component to initialize Firebase subscription
function FirebaseDataProvider({ children }: { children: React.ReactNode }) {
  useFirebaseData();
  return <>{children}</>;
}

function App() {

  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <FirebaseDataProvider>
            <Toaster />
            <Router />
          </FirebaseDataProvider>
        </TooltipProvider>
      </QueryClientProvider>
    </Provider>
  );
}

export default App;
