#!/usr/bin/env bash
# Regenerate the bundled C2PA development signing credentials in
# c2pa_dev_cert.py. These are deliberately untrusted — see that module's
# docstring. Run from the repository root:  ./scripts/make_dev_cert.sh
#
# The certificate profile is not arbitrary; c2pa-python rejects anything else:
#   - ECDSA P-256 ("es256"); the private key must be PKCS#8, not SEC1
#   - a chain (leaf + CA), not a bare self-signed leaf
#   - leaf: CA:FALSE, keyUsage digitalSignature, EKU emailProtection
set -euo pipefail

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
cd "$workdir"

cat > ca.cnf <<'EOF'
[req]
distinguished_name = dn
x509_extensions = v3
prompt = no
[dn]
CN = CrispTTS Development Root CA
O  = CrispTTS
OU = Untrusted Development Certificate
C  = DE
[v3]
basicConstraints = critical, CA:TRUE, pathlen:0
keyUsage = critical, keyCertSign, cRLSign
subjectKeyIdentifier = hash
EOF

cat > leaf.cnf <<'EOF'
[req]
distinguished_name = dn
prompt = no
[dn]
CN = CrispTTS Development Signer
O  = CrispTTS
OU = Untrusted Development Certificate
C  = DE
[v3]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, nonRepudiation
extendedKeyUsage = emailProtection
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer
EOF

openssl ecparam -name prime256v1 -genkey -noout -out ca.key
openssl req -new -x509 -key ca.key -out ca.pem -days 3650 -sha256 -config ca.cnf
openssl ecparam -name prime256v1 -genkey -noout -out leaf.key
openssl req -new -key leaf.key -out leaf.csr -config leaf.cnf
openssl x509 -req -in leaf.csr -CA ca.pem -CAkey ca.key -CAcreateserial \
    -out leaf.pem -days 3650 -sha256 -extfile leaf.cnf -extensions v3
openssl pkcs8 -topk8 -nocrypt -in leaf.key -out leaf_pkcs8.key
cat leaf.pem ca.pem > chain.pem

echo "Generated. Paste chain.pem and leaf_pkcs8.key into c2pa_dev_cert.py:"
echo "  $workdir/chain.pem"
echo "  $workdir/leaf_pkcs8.key"
cp chain.pem leaf_pkcs8.key "${OLDPWD}/" && echo "(copied to $OLDPWD)"
