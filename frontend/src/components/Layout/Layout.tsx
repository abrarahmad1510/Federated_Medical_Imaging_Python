import React, { ReactNode, useState } from 'react';
import { styled } from '@mui/material/styles';
import {
AppBar,
Box,
CssBaseline,
Drawer,
IconButton,
List,
ListItem,
ListItemIcon,
ListItemText,
Toolbar,
Typography,
useMediaQuery,
useTheme,
} from '@mui/material';
import {
Menu as MenuIcon,
Dashboard as DashboardIcon,
Science as ScienceIcon,
ModelTraining as ModelTrainingIcon,
People as PeopleIcon,
Analytics as AnalyticsIcon,
Settings as SettingsIcon,
Logout as LogoutIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
const drawerWidth = 240;
const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })<{
open?: boolean;
}>(({ theme, open }) => ({
flexGrow: 1,
padding: theme.spacing(3),
transition: theme.transitions.create('margin', {
easing: theme.transitions.easing.sharp,
duration: theme.transitions.duration.leavingScreen,
}),
marginLeft: `-${drawerWidth}px`,
...(open && {
transition: theme.transitions.create('margin', {
easing: theme.transitions.easing.easeOut,
duration: theme.transitions.duration.enteringScreen,
}),
marginLeft: 0,
}),
}));
const DrawerHeader = styled('div')(({ theme }) => ({
display: 'flex',
alignItems: 'center',
padding: theme.spacing(0, 1),
...theme.mixins.toolbar,
justifyContent: 'flex-end',
}));
interface LayoutProps {
children: ReactNode;
}
const menuItems = [
{ text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
{ text: 'Experiments', icon: <ScienceIcon />, path: '/experiments' },
{ text: 'Models', icon: <ModelTrainingIcon />, path: '/models' },
{ text: 'Clients', icon: <PeopleIcon />, path: '/clients' },
{ text: 'Monitoring', icon: <AnalyticsIcon />, path: '/monitoring' },
{ text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
];
const Layout: React.FC<LayoutProps> = ({ children }) => {
const theme = useTheme();
const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
const [mobileOpen, setMobileOpen] = useState(false);
const [desktopOpen, setDesktopOpen] = useState(true);
const navigate = useNavigate();
const location = useLocation();
const { user, logout } = useAuth();
const handleDrawerToggle = () => {
if (isMobile) {
setMobileOpen(!mobileOpen);
} else {
setDesktopOpen(!desktopOpen);
}
};
const handleNavigation = (path: string) => {
navigate(path);
if (isMobile) {
setMobileOpen(false);
}
};
const handleLogout = () => {
logout();
navigate('/login');
};
const drawer = (
<div>
<Toolbar>
<Typography variant="h6" noWrap sx={{ flexGrow: 1 }}>
Medical FL Platform
</Typography>
</Toolbar>
<Box sx={{ px: 2, py: 1 }}>
<Typography variant="subtitle2" color="textSecondary">
{user?.email}
</Typography>
<Typography variant="caption" color="textSecondary">
{user?.is_admin ? 'Administrator' : 'Researcher'}
</Typography>
</Box>
<List>
{menuItems.map((item) => (
<ListItem
button
key={item.text}
onClick={() => handleNavigation(item.path)}
selected={location.pathname === item.path}
sx={{
borderRadius: 1,
mb: 0.5,
'&.Mui-selected': {
backgroundColor: theme.palette.primary.light + '20',
'&:hover': {
backgroundColor: theme.palette.primary.light + '30',
},
},
}}
>
<ListItemIcon sx={{ color: 'inherit' }}>
{item.icon}
</ListItemIcon>
<ListItemText primary={item.text} />
</ListItem>
))}
<ListItem
button
onClick={handleLogout}
sx={{
borderRadius: 1,
mt: 2,
color: theme.palette.error.main,
'&:hover': {
backgroundColor: theme.palette.error.light + '20',
},
}}
>
<ListItemIcon sx={{ color: 'inherit' }}>
<LogoutIcon />
</ListItemIcon>
<ListItemText primary="Logout" />
</ListItem>
</List>
</div>
);
return (
<Box sx={{ display: 'flex' }}>
<CssBaseline />
<AppBar
position="fixed"
sx={{
zIndex: theme.zIndex.drawer + 1,
...(desktopOpen && !isMobile && { width: `calc(100% - ${drawerWidth}px)` }),
}}
>
<Toolbar>
<IconButton
color="inherit"
aria-label="open drawer"
edge="start"
onClick={handleDrawerToggle}
sx={{ mr: 2 }}
>
<MenuIcon />
</IconButton>
<Typography variant="h6" noWrap sx={{ flexGrow: 1 }}>
{menuItems.find(item => item.path === location.pathname)?.text || "Medical FL Platform"}
{menuItems.find(item => item.path === location.pathname)?.text || "Medical FL Platform"}
</Typography>
<Box sx={{ display: 'flex', alignItems: 'center' }}>
<Typography variant="body2" sx={{ mr: 2 }}>
v1.0.0
</Typography>
</Box>
</Toolbar>
</AppBar>
{isMobile ? (
<Drawer
variant="temporary"
open={mobileOpen}
onClose={handleDrawerToggle}
ModalProps={{ keepMounted: true }}
sx={{
'& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
}}
>
{drawer}
</Drawer>
) : (
<Drawer
variant="persistent"
open={desktopOpen}
sx={{
width: drawerWidth,
flexShrink: 0,
'& .MuiDrawer-paper': {
width: drawerWidth,
boxSizing: 'border-box',
},
}}
>
{drawer}
</Drawer>
)}
<Main open={desktopOpen && !isMobile}>
<DrawerHeader />
<Box sx={{ mt: 2 }}>
{children}
</Box>
</Main>
</Box>
);
};
export default Layout;
