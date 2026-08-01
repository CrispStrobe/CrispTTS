"""Bundled C2PA development signing credentials for CrispTTS.

CrispTTS signs every C2PA-capable output by default, so that AI provenance
travels with the file in an interoperable, cryptographically verifiable form
rather than only as a CrispTTS-specific watermark. Signing needs a credential,
and requiring every user to obtain one would mean the default install signs
nothing — so these are bundled.

**These credentials are deliberately untrusted.** The private key below is
public: it is in the source tree, in the wheel, and in every checkout. A
manifest signed with it proves only that the file has not been altered since
signing. It does NOT attest to who produced the file, and it will NOT validate
against the C2PA known-certificate trust list.

For a credential that others can verify as yours, obtain a certificate from a
C2PA-recognised authority and pass ``--c2pa-cert`` / ``--c2pa-key`` (or set
``C2PA_CERT_PATH`` / ``C2PA_KEY_PATH``). :func:`watermark.c2pa_sign_file_ex`
reports which of the two was used, and CrispTTS never describes a bundled-cert
signature as trusted.

Profile: ECDSA P-256 (es256), leaf + development root CA, EKU emailProtection,
valid until 2036. Regenerate with ``scripts/make_dev_cert.sh``.
"""

#: Leaf signing certificate followed by its development root CA, PEM-encoded.
DEV_CERT_CHAIN_PEM = """\
-----BEGIN CERTIFICATE-----
MIICXDCCAgKgAwIBAgIURNCVOFtu55zSjLPbE1aN0Yb1iWAwCgYIKoZIzj0EAwIw
czElMCMGA1UEAwwcQ3Jpc3BUVFMgRGV2ZWxvcG1lbnQgUm9vdCBDQTERMA8GA1UE
CgwIQ3Jpc3BUVFMxKjAoBgNVBAsMIVVudHJ1c3RlZCBEZXZlbG9wbWVudCBDZXJ0
aWZpY2F0ZTELMAkGA1UEBhMCREUwHhcNMjYwODAxMTU1ODE2WhcNMzYwNzI5MTU1
ODE2WjByMSQwIgYDVQQDDBtDcmlzcFRUUyBEZXZlbG9wbWVudCBTaWduZXIxETAP
BgNVBAoMCENyaXNwVFRTMSowKAYDVQQLDCFVbnRydXN0ZWQgRGV2ZWxvcG1lbnQg
Q2VydGlmaWNhdGUxCzAJBgNVBAYTAkRFMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcD
QgAE2Ju4Bf7WybRqT+TmYbAdpU8SqSgVZPEbMxODG3bBFx9LKc5dn1um3m935qUE
axo+QfT0ynOSntZgKfGiHfHReaN1MHMwDAYDVR0TAQH/BAIwADAOBgNVHQ8BAf8E
BAMCBsAwEwYDVR0lBAwwCgYIKwYBBQUHAwQwHQYDVR0OBBYEFFyJ0YqUUNGwfYP+
WdPdZpZ5XEOeMB8GA1UdIwQYMBaAFDwybjBUjTSSe+T74qaviZtOjHuFMAoGCCqG
SM49BAMCA0gAMEUCIAIS9bu+inuZZEkxPJjxHOYGtKk9f77GSYq+VELG7SdSAiEA
0pv/KyCO5NPbph5gOEnNeXxIjkd72I2XcI49jo0UwDw=
-----END CERTIFICATE-----
-----BEGIN CERTIFICATE-----
MIICLTCCAdOgAwIBAgIUIV/KRJHW2sHVlIKTRWh4oY82qWMwCgYIKoZIzj0EAwIw
czElMCMGA1UEAwwcQ3Jpc3BUVFMgRGV2ZWxvcG1lbnQgUm9vdCBDQTERMA8GA1UE
CgwIQ3Jpc3BUVFMxKjAoBgNVBAsMIVVudHJ1c3RlZCBEZXZlbG9wbWVudCBDZXJ0
aWZpY2F0ZTELMAkGA1UEBhMCREUwHhcNMjYwODAxMTU1ODE0WhcNMzYwNzI5MTU1
ODE0WjBzMSUwIwYDVQQDDBxDcmlzcFRUUyBEZXZlbG9wbWVudCBSb290IENBMREw
DwYDVQQKDAhDcmlzcFRUUzEqMCgGA1UECwwhVW50cnVzdGVkIERldmVsb3BtZW50
IENlcnRpZmljYXRlMQswCQYDVQQGEwJERTBZMBMGByqGSM49AgEGCCqGSM49AwEH
A0IABG5x6RWbdohuiOqYa8U2sGpTKNsCmReutWUgIUpEhsnaPb/aUR3hPOkDJPfM
e0+7EbPRakhZLeOS/CAnOSr9f6ejRTBDMBIGA1UdEwEB/wQIMAYBAf8CAQAwDgYD
VR0PAQH/BAQDAgEGMB0GA1UdDgQWBBQ8Mm4wVI00knvk++Kmr4mbTox7hTAKBggq
hkjOPQQDAgNIADBFAiEAtfN/eC+e29PLoJ1JBhtoBRyVy5VjlHroa/jjUN5UkhoC
IBUXu9a5SMzMTikS+F/Ar3FIn3Vc5x4hNyRS3XzHCKrI
-----END CERTIFICATE-----
"""

#: PKCS#8 private key for the leaf certificate. Public by design — see above.
DEV_PRIVATE_KEY_PEM = """\
-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgfIYagW8VLZBtlMus
ju/ZdTtGSF6H45NwZP0/sRHBssuhRANCAATYm7gF/tbJtGpP5OZhsB2lTxKpKBVk
8RszE4MbdsEXH0spzl2fW6beb3fmpQRrGj5B9PTKc5Ke1mAp8aId8dF5
-----END PRIVATE KEY-----
"""
