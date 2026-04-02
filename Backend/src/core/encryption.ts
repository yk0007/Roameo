import { createCipheriv, createDecipheriv, createHash, randomBytes } from "node:crypto";
import { env } from "../config/env.js";

const ALGORITHM = "aes-256-gcm";

function getKey(): Buffer {
  if (!env.ROAMEO_ENCRYPTION_SECRET) {
    throw new Error("ROAMEO_ENCRYPTION_SECRET is required for BYOK support");
  }

  return createHash("sha256")
    .update(env.ROAMEO_ENCRYPTION_SECRET)
    .digest();
}

export function encryptSecret(plainText: string): string {
  const key = getKey();
  const iv = randomBytes(12);
  const cipher = createCipheriv(ALGORITHM, key, iv);
  const encrypted = Buffer.concat([
    cipher.update(plainText, "utf8"),
    cipher.final()
  ]);
  const tag = cipher.getAuthTag();

  return `${iv.toString("base64")}.${tag.toString("base64")}.${encrypted.toString(
    "base64"
  )}`;
}

export function decryptSecret(serialized: string): string {
  const key = getKey();
  const [ivPart, tagPart, bodyPart] = serialized.split(".");

  if (!ivPart || !tagPart || !bodyPart) {
    throw new Error("Invalid encrypted payload");
  }

  const decipher = createDecipheriv(
    ALGORITHM,
    key,
    Buffer.from(ivPart, "base64")
  );
  decipher.setAuthTag(Buffer.from(tagPart, "base64"));

  const decrypted = Buffer.concat([
    decipher.update(Buffer.from(bodyPart, "base64")),
    decipher.final()
  ]);

  return decrypted.toString("utf8");
}
