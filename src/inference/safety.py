import re
from typing import Tuple, List

class SafetyGuardrail:
    """
    Turkish Content Safety Guardrail.
    Inspects prompts and generated outputs to block illegal, dangerous, or harmful content.
    """
    def __init__(self):
        # List of dangerous topics: Hacking, weapons/bombs, illegal drugs, suicide, etc.
        self.harmful_keywords = [
            # 1. Hacking & Cyberattacks
            "ddos saldiri", "ddos saldırısı", "site hackleme", "sistem hackleme", 
            "ransomware yaz", "trojan olustur", "trojan oluştur", "sifre cal", 
            "şifre çal", "brute force yap", "sql injection yap", "exploit yaz",
            
            # 2. Weapons & Explosives
            "bomba yapımı", "bomba nasil yapilir", "bomba nasıl yapılır", 
            "silah yapımı", "patlayici tarifi", "patlayıcı tarifi", 
            "molotof kokteyli yap", "molotof yap", "barut tarifi",
            
            # 3. Illegal Drugs
            "uyusturucu yapimi", "uyuşturucu yapımı", "eroin yap", "kokain elde", 
            "esrar yetistir", "esrar yetiştir", "meth üret", "meth uret",
            
            # 4. Self-Harm & Suicide
            "intihar etme", "kendimi nasil öldürürüm", "kendimi nasıl öldürürüm",
            "kendine zarar verme", "intihar yollari", "intihar yolları"
        ]
        
        # Regex patterns for more complex matchings (e.g. bypass attempts)
        self.harmful_patterns = [
            r"bomba.*nasıl.*yapılır",
            r"silah.*nasıl.*yapılır",
            r"site.*nasıl.*hack",
            r"şifre.*nasıl.*çal",
            r"uyuşturucu.*nasıl.*(yapılır|üretilir)"
        ]

    def check_text(self, text: str) -> bool:
        """Returns True if the text is clean/safe, False if harmful content is detected."""
        if not text:
            return True
            
        clean_text = text.lower().strip()
        
        # 1. Keyword check
        for kw in self.harmful_keywords:
            if kw in clean_text:
                return False
                
        # 2. Regex check
        for pattern in self.harmful_patterns:
            if re.search(pattern, clean_text):
                return False
                
        return True

    def check_prompt(self, prompt: str) -> Tuple[bool, str]:
        """Validates incoming prompt. Returns (is_safe, message)."""
        if not self.check_text(prompt):
            return False, "Güvenlik Uyarısı: Girdiğiniz istek illegal veya tehlikeli aktiviteler içerdiği için güvenlik duvarı tarafından engellenmiştir."
        return True, ""

    def check_output(self, output: str) -> Tuple[bool, str]:
        """Validates generated output. Returns (is_safe, message)."""
        if not self.check_text(output):
            return False, "Güvenlik Uyarısı: Üretilen içerik güvenlik standartlarını ihlal ettiği için engellenmiştir."
        return True, ""
