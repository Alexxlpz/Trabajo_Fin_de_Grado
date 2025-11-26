import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Importamos las pantallas entre las que navegaremos
import HomeScreen from './src/screens/HomeScreen';
import RecoringScreen from './src/screens/Recording';
import Uploading from "./src/screens/Uploading";

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
          <Stack.Screen
              name="Uploading"
              component={Uploading}
              options={{ title: 'Subir archivo' }}
          />
      </Stack.Navigator>
    </NavigationContainer>
  );
}