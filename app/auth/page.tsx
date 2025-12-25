"use client";

import { SignIn } from "@clerk/nextjs";

export default function AuthPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="w-full max-w-md p-8 bg-white shadow rounded">
        <SignIn
          path="/auth"       // URL of this page
          routing="path"     // Use path-based routing
          redirectUrl="/"    // Where to go after sign-in/sign-up
        />
      </div>
    </div>
  );
}
