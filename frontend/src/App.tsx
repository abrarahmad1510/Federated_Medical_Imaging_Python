import React from 'react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { BrowserRouter as Router } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { SnackbarProvider } from 'notistack';
import { Provider } from 'react-redux';
import theme from './theme';
import { store } from './store';
import { AuthProvider } from './contexts/AuthContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import Layout from './components/Layout/Layout';
import Routes from './routes';
const queryClient = new QueryClient({
defaultOptions: {
queries: {
refetchOnWindowFocus: false,
retry: 1,
staleTime: 5 * 60 * 1000, // 5 minutes
},
},
});
function App() {
return (
<ThemeProvider theme={theme}>
<CssBaseline />
<QueryClientProvider client={queryClient}>
<Provider store={store}>
<SnackbarProvider
maxSnack={3}
anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
autoHideDuration={3000}
>
<AuthProvider>
<WebSocketProvider>
<Router>
<Layout>
<Routes />
</Layout>
</Router>
</WebSocketProvider>
</AuthProvider>
</SnackbarProvider>
</Provider>
<ReactQueryDevtools initialIsOpen={false} />
</QueryClientProvider>
</ThemeProvider>
);
}
export default App;
