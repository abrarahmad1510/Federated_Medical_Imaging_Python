import React from 'react';
import { Container, Typography, Box } from '@mui/material';

const Models: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Models
        </Typography>
        <Typography variant="body1">
          This is the Models page. Content will be added here.
        </Typography>
      </Box>
    </Container>
  );
};

export default Models;
