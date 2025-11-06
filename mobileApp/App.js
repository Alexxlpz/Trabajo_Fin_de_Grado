import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Importamos las pantallas entre las que navegaremos
import HomeScreen from './src/screens/HomeScreen';
import RecoringScreen from './src/screens/Recording';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ title: 'Inicio' }}
        />
        <Stack.Screen
          name="Recording"
          component={RecoringScreen}
          //options={{ headerShown: false }}¿¿¿???
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}