import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Platform, StatusBar } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useSession } from '../SessionContext';

export default function NavBar({ navigation, back, options }: any) {
    const {isLoggedIn} = useSession();
    const title = options?.title ?? 'AGRODOC';
    const showProfile = options?.showProfile !== false;
    const showBrand = options?.showBrand !== false;
    const showBack = !!back;

    return (
        <View style={styles.container}>
            <View style={styles.content}>
                <View style={styles.sideArea}>
                    {showBack ? (
                        <TouchableOpacity
                            onPress={() => navigation.goBack()}
                            activeOpacity={0.75}
                            style={styles.iconButton}
                        >
                            <Ionicons name="chevron-back" size={24} color="#FFFFFF" />
                        </TouchableOpacity>
                    ) : showBrand ? (
                        <View style={styles.brandPill}>
                            <Ionicons name="leaf" size={14} color="#E9FFF4" />
                            <Text style={styles.brandText}>AGRODOC</Text>
                        </View>
                    ) : (
                        <View style={styles.iconPlaceholder} />
                    )}
                </View>

                <Text numberOfLines={1} ellipsizeMode="tail" style={styles.title}>{title}</Text>

                <View style={[styles.sideArea, styles.rightArea]}>
                    {showProfile ? (
                        <TouchableOpacity
                            activeOpacity={0.75}
                            onPress={() => { isLoggedIn ? navigation.navigate('Profile') : navigation.navigate('Login')}}
                            style={styles.iconButton}
                        >
                            {isLoggedIn ? (
                                <Ionicons name="person-circle-outline" size={32} color="#FFFFFF" />
                            ) : (
                                <Ionicons name="log-in-outline" size={32} color="#FFFFFF" />
                            )}
                        </TouchableOpacity>
                    ) : (
                        <View style={styles.iconPlaceholder} />
                    )}
                </View>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        backgroundColor: '#00875A',
        width: '100%',
        alignSelf: 'stretch',
        paddingTop: Platform.OS === 'ios' ? 62 : (StatusBar.currentHeight ? StatusBar.currentHeight + 12 : 18),
        paddingBottom: 12,
        paddingHorizontal: 18,
        borderBottomLeftRadius: 0,
        borderBottomRightRadius: 0,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.16,
        shadowRadius: 18,
        elevation: 10,
        overflow: 'hidden',
    },
    content: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        minHeight: 38,
        width: '100%',
    },
    sideArea: {
        minWidth: 56,
        maxWidth: 96,
        alignItems: 'flex-start',
        justifyContent: 'center',
    },
    rightArea: {
        alignItems: 'flex-end',
    },
    title: {
        flex: 1,
        textAlign: 'center',
        color: '#FFFFFF',
        fontSize: 19,
        fontWeight: '800',
        letterSpacing: 0.3,
        paddingHorizontal: 10,
        includeFontPadding: false,
    },
    iconButton: {
        width: 42,
        height: 42,
        borderRadius: 21,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'transparent',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.28)',
    },
    brandPill: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 12,
        height: 36,
        borderRadius: 18,
        backgroundColor: 'rgba(255,255,255,0.16)',
        gap: 6,
    },
    brandText: {
        color: '#FFFFFF',
        fontSize: 13,
        fontWeight: '800',
        letterSpacing: 1,
    },
    iconPlaceholder: {
        width: 42,
        height: 42,
    },
});