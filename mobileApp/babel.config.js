module.exports = function(api) {
    api.cache(true);
    return {
        presets: ['babel-preset-expo'], // Preset correcto para Expo
        plugins: [
            // Elimina el transform-private-methods, ya viene en el preset de expo
            ['module:react-native-dotenv', {
                "moduleName": "@env",
                "path": ".env",
                "safe": false,
                "allowUndefined": true
            }]
        ],
    };
};