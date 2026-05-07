import React from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';

// Importamos las pantallas entre las que navegaremos
import HomeScreen from './src/screens/HomeScreen';
import RecoringScreen from './src/screens/Recording';
import Uploading from "./src/screens/Uploading";
import Login from "./src/screens/Login";
import Register from "./src/screens/Register";
import { SessionProvider } from './src/SessionContext';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <SessionProvider>
      <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={({ navigation }) => ({
            headerStyle: { 
                backgroundColor: '#00875A',
                elevation: 0,
                shadowOpacity: 0,
            },
            headerTintColor: styles.headerText.color,
            headerTitleStyle: { fontWeight: 'bold' },
            headerTitleAlign: 'center',
            headerShadowVisible: true,
            headerRight: () => (
                <TouchableOpacity 
                    activeOpacity={0.7}
                    onPress={() => navigation.navigate('Login')}
                >
                    <Ionicons name="person-circle-outline" size={38} color="white" />
                </TouchableOpacity>
            ),
        })}
      >
          <Stack.Screen
              name="Home"
              component={HomeScreen}
              options={{ title: 'Inicio' }}
          />
          <Stack.Screen
              name="Recording"
              component={RecoringScreen}
              options={{ title: 'Grabar' }}
          />
          <Stack.Screen
              name="Uploading"
              component={Uploading}
              options={{ title: 'Subiendo' }}
          />
          <Stack.Screen
              name="Login"
              component={Login}
              options={{ title: 'Iniciar Sesión' }}
          />
          <Stack.Screen
              name="Register"
              component={Register}
              options={{ title: 'Crear Cuenta' }}
          />
      </Stack.Navigator>
    </NavigationContainer>
    </SessionProvider>
  );
}

const styles = {
    header: {
        backgroundColor: '#00875A', // Color verde oscuro de la barra
        paddingTop: 45, // Espacio para la barra de estado del móvil
        paddingBottom: 15,
        paddingHorizontal: 20,
    },
    headerText: {
        color: '#FFFFFF',
        fontSize: 18,
        fontWeight: 'bold',
    }
}