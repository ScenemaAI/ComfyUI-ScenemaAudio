# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Preset library for Scenema Audio Generate.

Ported directly from the official blog demos at scenema.ai/audio.
Each preset is a byte-for-byte port of a demo XML prompt from the
announcement article, translated into our field-based format.

Keep this in sync with web/js/scenema_presets.js.
"""

CUSTOM = "Custom"

PRESETS = {
    "Old Male Storyteller (fireside)": {
        "voice_description": "Male, mid 60s. Deep baritone with gravel. Slight Southern American inflection. Worn but warm. The voice of someone who has seen too much and chosen kindness anyway. Nostalgic, firelight cadence.",
        "gender": "male",
        "scene": "Quiet indoor room",
        "custom_scene": "Fireside, night, crickets in the distance",
        "action_tags": "He settles into his chair and stares at the fire",
        "speech_text": "There was a summer, back when the river still ran clear, when my father took me out past the property line and pointed at the stars. He said, boy, every one of those is a story somebody forgot to write down. [He smiles to himself] I have been writing them down ever since.",
    },
    "Young Woman (breathless discovery)": {
        "voice_description": "Female, early 20s. Bright soprano. Slightly breathy. American West Coast. The kind of voice that smiles while speaking. Breathless awe, tumbling over words.",
        "gender": "female",
        "scene": "Outdoor, open air",
        "custom_scene": "An open field, something glowing in front of her",
        "action_tags": "She freezes, eyes wide",
        "speech_text": "Oh my god. Oh my god, it is real. I thought they were lying, I thought it was just some internet thing but it is actually here and it is glowing and I do not know what to do with my hands right now.",
    },
    "Terrified Whisper": {
        "voice_description": "Male, mid 30s. Whisper. Terrified. Shaking. A man hiding, trying not to be found. Every word is a risk. Breath catching between words.",
        "gender": "male",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "He presses against the wall, barely breathing",
        "speech_text": "Listen to me. Do not turn around. The man in the grey coat has been following us since the bridge. I need you to walk to the cafe on the corner, order something, and leave through the back. I will find you. Do you understand? Nod if you understand.",
    },
    "Irish Woman, Dry Wit": {
        "voice_description": "Woman, mid 40s. Strong Irish accent, Dublin. Dry, sardonic, cutting. Bone-dry wit. She sounds like she has seen it all and finds most of it beneath her.",
        "gender": "female",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "She speaks flatly, unimpressed",
        "speech_text": "Apparently the committee has decided that what this building really needs is another meeting room. Because the problem with this organization was never the decisions. It was that we did not have enough places to avoid making them.",
    },
    "Rage to Vulnerability": {
        "voice_description": "A man on the edge. Explosive rage building with every sentence. Gravelly, intimidating. Italian-American inflection. Controlled fury that could snap at any moment. The kind of anger that comes from deep disrespect.",
        "gender": "male",
        "scene": "Quiet indoor room",
        "custom_scene": "A dimly lit office, late at night",
        "action_tags": "He stands up slowly, voice dangerously low",
        "speech_text": "You come into my house, you eat my food, and then you got the nerve to tell me how to run my business. You know what your problem is? You got no respect. None. Zero. [Voice rising, finger pointing] I built this thing from nothing, nothing, while you were sitting on your ass doing God knows what. So don't come in here with that attitude. You understand me?",
    },
    "Eulogy (Aeschylus, grief)": {
        "voice_description": "Woman, mid 60s. Deep. Extremely slow. Heavy with grief. Each word lands like a stone dropped into still water. Long pauses between phrases. Barely above a whisper.",
        "gender": "female",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "She speaks so slowly that each phrase feels like its own sentence. Heavy pauses. The weight of loss in every breath.",
        "speech_text": "Even in our sleep. Pain which cannot forget. Falls drop by drop upon the heart. Until in our own despair. Against our will. Comes wisdom. Through the awful grace of God.",
    },
    "Terror (sobbing)": {
        "voice_description": "Woman, late 20s. Voice shaking violently. Hyperventilating. Sobbing. Choking on tears. Words barely coming out between gasps for air. Throat tight with panic. Speaking through crying.",
        "gender": "female",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "She gasps for air between sobs, voice breaking on every word, barely able to speak through the tears",
        "speech_text": "Please. Please help me. I can hear them downstairs. They broke the window. My baby is with me. Please send help. Please hurry. Please.",
    },
    "Villain (laughing menace)": {
        "voice_description": "Male. Deep, resonant, theatrical voice dripping with contempt and dark amusement. Dramatic pauses. Shifting between sinister whispers and booming declarations.",
        "gender": "male",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "He laughs, quiet at first, then louder, then speaks with cold precision",
        "speech_text": "Heheheh. Hahahaha! Oh I have waited so long for this. They told me you were clever. They said be careful. And here you are, on your knees, with nothing left. Tell me. Was it worth it? All that running?",
    },
    "Rain and Thunder (SFX)": {
        "voice_description": "Male, mid 40s. Baritone. Weathered. Urgent, projecting over wind and rain.",
        "gender": "male",
        "scene": "Rainy outdoors",
        "custom_scene": "Open dock in a thunderstorm, heavy rain, waves crashing against the pier",
        "action_tags": "Heavy rain and wind howling. He cups his hands and shouts over the wind and rain",
        "speech_text": "Get the lines! Get the lines now! She is pulling loose! If we lose this boat we lose everything! [Thunder cracks overhead] [He screams louder] Move! I said move!",
    },
    "Italian Cooking Show (SFX)": {
        "voice_description": "Female, mid 30s. Warm, enthusiastic. Italian accent. A home cook who treats every meal like a celebration.",
        "gender": "female",
        "scene": "Café or restaurant",
        "custom_scene": "Busy home kitchen, oil sizzling in a hot pan, pots bubbling on the stove",
        "action_tags": "Oil sizzling loudly in a hot pan, a pot bubbling on the stove. She talks over the sizzling, gesturing with a wooden spoon, energetic and happy",
        "speech_text": "Okay now this is the important part. You wait until the oil is really hot, you see the smoke? That is when you drop the garlic in. [Garlic hits the hot oil with a loud sizzle and crackle] [She stirs quickly, laughing] Beautiful! You smell that? Now we add the tomatoes and let it all come together.",
    },
    "Kid Explaining Dinosaurs": {
        "voice_description": "Boy, 8 years old. Small clear voice. Speaking carefully like he is the authority on this subject. A child explaining something important to someone younger.",
        "gender": "male",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "He speaks seriously, like a tiny professor",
        "speech_text": "Okay so dinosaurs. They were really really big, like bigger than this whole house. And they lived a million billion years ago. And you know what happened? A giant rock came from space and hit the earth and then it got really cold and they all had to go away. But birds are actually dinosaurs. So technically we have dinosaurs right now.",
    },
    "British Woman, East London Rage": {
        "voice_description": "Shrill angry British female voice, East London accent. Screaming and furious.",
        "gender": "female",
        "scene": "Absolute silence",
        "custom_scene": "A messy flat, pointing at the camera",
        "action_tags": "She points at the camera, face twisted with rage",
        "speech_text": "Are you having a bloody laugh? You absolute muppet! I told you THREE times to sort the bins out and what do I come home to? This! This absolute disaster! I swear to God if you don't get your shit together by tomorrow I am DONE. Finished! Pack your bags and piss off back to your mum's! I am NOT joking!",
    },
}

PRESET_NAMES = [CUSTOM] + list(PRESETS.keys())
