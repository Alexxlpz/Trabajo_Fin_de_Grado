import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Alert,
  Image,
} from 'react-native';
import { IP_ADDRESS } from "@env";
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useSession } from '../SessionContext';
import { useNavigation } from '@react-navigation/native';
import { User } from '../classes/User';
import AppBackground from '../component/AppBackground';

type RootStackParamList = {
  Home: undefined;
  Login: undefined;
  Register: undefined;
};

type LoginScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>;


export default function LoginScreen() {
  const navigation = useNavigation<LoginScreenNavigationProp>();
  const { setRecents, setIsLoggedIn, setUser } = useSession();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

    const handleLogin = async (email: string, password: string) => {
            try {
                const response = await fetch(`http://${IP_ADDRESS}:8000/login`, {
                  method: 'POST',
                  headers: {
                      'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({ 
                    email: email,
                    password: password
                   }),
                });
              
              if (!response.ok) {
                  Alert.alert('Error', `La respuesta del servidor no es correcta (Status: ${response.status})`);
                  return;
              }

              const jsonRecived = await response.json();
               console.log('Respuesta del servidor recibida');

                if (jsonRecived && jsonRecived.message && Array.isArray(jsonRecived.recent_list)) {
                    // guardo los path de manera local para tenerlos disponibles en la pantalla de upload y redirijo a la pantalla anterior
                    setRecents(jsonRecived.recent_list);
                    setIsLoggedIn(true);
                    setUser(jsonRecived.user);

                    navigation.reset({
                        index: 0,
                        routes: [ { name: 'Home' } ],
                    });
                    
                } else {
                    Alert.alert('Error', 'Respuesta inesperada del servidor');
                    console.error('Respuesta inválida del servidor:', jsonRecived);
                }
            } catch (error) {
              console.error(error);
              Alert.alert('Error de conexión', 'No se pudo conectar con el servidor.');
            }
    }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <AppBackground>
        <ScrollView contentContainerStyle={[styles.overlay, styles.scrollContent]}>
          <View style={styles.logoContainer}>
            <View style={styles.logo}>
              <Image source={require('../../assets/icon.png')} style={styles.logoImage} />
            </View>
            <Text style={styles.title}>AGRODOC</Text>
            <Text style={styles.subtitle}>Una aplicación de agricultores para agricultores</Text>
          </View>

          <View style={styles.formContainer}>
            <View style={styles.inputGroup}>
              <Text style={styles.label}>Correo electrónico</Text>
              <View style={styles.inputWrapper}>
                <TextInput
                  style={styles.input}
                  placeholder="ejemplo@habichuela.com"
                  placeholderTextColor="#999"
                  value={email}
                  onChangeText={setEmail}
                  keyboardType="email-address"
                />
              </View>
            </View>

            <View style={styles.inputGroup}>
              <View style={styles.passwordHeader}>
                <Text style={styles.label}>Contraseña</Text>
                <TouchableOpacity onPress={() => Alert.alert('Función no disponible', 'Esta funcionalidad se añadirá en futuras actualizaciones de la aplicación')}>
                  <Text style={styles.forgotPassword}>¿Olvidaste tu contraseña?</Text>
                </TouchableOpacity>
              </View>
              <View style={styles.inputWrapper}>
                <TextInput
                  style={styles.input}
                  placeholder="••••••••"
                  placeholderTextColor="#999"
                  value={password}
                  onChangeText={setPassword}
                  secureTextEntry={true}
                />
              </View>
            </View>

            <TouchableOpacity
              style={styles.loginButton}
              onPress={() => handleLogin(email, password)}
            >
              <Text style={styles.loginButtonText}>Log In →</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.registerButton}
              onPress={() => navigation.navigate('Register')}
              activeOpacity={0.85}
            >
              <Text style={styles.registerButtonText}>Crear una cuenta nueva</Text>
            </TouchableOpacity>

            <View style={styles.footer}>
              <Text style={styles.footerText}>
                Need access?{' '}
                <TouchableOpacity onPress={() => Alert.alert('Función no disponible', 'Esta funcionalidad se añadirá en futuras actualizaciones de la aplicación')}>
                  <Text style={styles.contactLink}>Contact System Admin</Text>
                </TouchableOpacity>
              </Text>
            </View>
          </View>
        </ScrollView>
      </AppBackground>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
      backgroundColor: 'transparent',
  },
  overlay: {
        flex: 1,
        backgroundColor: 'rgba(255, 255, 255, 0.65)', // Superposición blanca translúcida
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingHorizontal: 20,
    paddingVertical: 40,
  },
  logoContainer: {
    alignItems: 'center',
    marginBottom: 50,
  },
  logo: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#1B9A6E',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  logoText: {
    fontSize: 32,
    color: '#fff',
  },
  logoImage: {
    width: 36,
    height: 36,
    resizeMode: 'contain',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
    letterSpacing: 1,
  },
  subtitle: {
    fontSize: 12,
    color: '#999',
  },
  formContainer: {
    gap: 20,
  },
  inputGroup: {
    gap: 8,
  },
  label: {
    fontSize: 12,
    fontWeight: '600',
    color: '#333',
  },
  passwordHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderRadius: 8,
    paddingHorizontal: 12,
    height: 48,
  },
  icon: {
    fontSize: 18,
    marginRight: 10,
    color: '#999',
  },
  input: {
    flex: 1,
    fontSize: 14,
    color: '#333',
    paddingVertical: 12,
  },
  forgotPassword: {
    fontSize: 12,
    color: '#1B9A6E',
    fontWeight: '500',
  },
  loginButton: {
    backgroundColor: '#1B9A6E',
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
    marginTop: 10,
  },
  loginButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  registerButton: {
    borderWidth: 1,
    borderColor: '#1B9A6E',
    borderRadius: 8,
    paddingVertical: 13,
    alignItems: 'center',
    backgroundColor: '#EEF8F4',
  },
  registerButtonText: {
    color: '#167A57',
    fontSize: 15,
    fontWeight: '600',
  },
  footer: {
    alignItems: 'center',
    marginTop: 20,
  },
  footerText: {
    fontSize: 12,
    color: '#999',
  },
  contactLink: {
    color: '#333',
    fontWeight: '600',
  },
});
