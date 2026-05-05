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
              options={{
                  title: 'Home',
                  headerStyle: { backgroundColor: styles.header.backgroundColor },
                  headerTintColor: styles.headerText.color,
                  headerTitleStyle: { fontWeight: 'bold' },
                  headerTitleAlign: 'center',
                  headerShadowVisible: true,
              }}
          />
          <Stack.Screen
              name="Recording"
              component={RecoringScreen}
              options={{
                  headerStyle: { backgroundColor: styles.header.backgroundColor },
                  headerTintColor: styles.headerText.color,
                  headerTitleStyle: { fontWeight: 'bold' },
                  headerTitleAlign: 'center',
                  headerShadowVisible: true,
              }}
              //options={{ headerShown: false }}¿¿¿???
          />
          <Stack.Screen
              name="Uploading"
              component={Uploading}
              options={{
                  headerStyle: { backgroundColor: styles.header.backgroundColor },
                  headerTintColor: styles.headerText.color,
                  headerTitleStyle: { fontWeight: 'bold' },
                  headerTitleAlign: 'center',
                  headerShadowVisible: true,
              }}
          />
      </Stack.Navigator>
    </NavigationContainer>
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