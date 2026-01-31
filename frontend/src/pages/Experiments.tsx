import React from 'react';
import { Container, Typography, Box } from '@mui/material';

const Experiments: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Experiments
        </Typography>
        <Typography variant="body1">
          This is the Experiments page. Content will be added here.
        </Typography>
      </Box>
    </Container>
  );
};

export default Experiments;
