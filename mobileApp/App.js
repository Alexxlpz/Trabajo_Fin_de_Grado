import React from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Importamos las pantallas entre las que navegaremos
import HomeScreen from './src/screens/HomeScreen';
import RecoringScreen from './src/screens/Recording';
import Uploading from "./src/screens/Uploading";
import Login from "./src/screens/Login";
import Register from "./src/screens/Register";
import { SessionProvider } from './src/SessionContext';
import NavBar from './src/component/NavBar';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <SessionProvider>
      <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
            header: (props) => <NavBar {...props} />,
        }}
      >
          <Stack.Screen
              name="Home"
              component={HomeScreen}
              options={{ title: 'AGRODOC', showProfile: true, showBrand: false }}
          />
          <Stack.Screen
              name="Recording"
              component={RecoringScreen}
              options={{ title: 'Grabar', showProfile: true }}
          />
          <Stack.Screen
              name="Uploading"
              component={Uploading}
              options={{ title: 'Subir Multimedia', showProfile: true }}
          />
          <Stack.Screen
              name="Login"
              component={Login}
              options={{ title: 'Iniciar Sesión', showProfile: false }}
          />
          <Stack.Screen
              name="Register"
              component={Register}
              options={{ title: 'Crear Cuenta', showProfile: false }}
          />
      </Stack.Navigator>
    </NavigationContainer>
    </SessionProvider>
  );
}
