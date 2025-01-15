import { Form, FormField, FormLabel } from '@radix-ui/react-form';
import { Button, Card, Flex, Heading, Separator, TextField } from '@radix-ui/themes';
import _ from 'lodash';
import React, { useCallback, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Routes } from './routes';

type FormValue = number | '';
interface FormData {
  Body1: {
    x: FormValue;
    y: FormValue;
    z: FormValue;
    vx: FormValue;
    vy: FormValue;
    vz: FormValue;
    mass: FormValue;
  };
  Body2: {
    x: FormValue;
    y: FormValue;
    z: FormValue;
    vx: FormValue;
    vy: FormValue;
    vz: FormValue;
    mass: FormValue;
  };
}

const SimulateForm: React.FC = () => {
  const navigate = useNavigate();

  const [formData, setFormData] = useState<FormData>({
    Body1: { x: -0.73, y: 0, z: 0, vx: 0, vy: -0.0015, vz: 0, mass: 1 },
    Body2: { x: 60.34, y: 0, z: 0, vx: 0, vy: 0.13, vz: 0, mass: 0.0123 },
  });

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    let newValue: FormValue = value === '' ? '' : parseFloat(value);
    setFormData((prev) => _.set({ ...prev }, name, newValue));
  }, []);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      try {
        const response = await fetch('http://localhost:8000/simulation', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(formData),
        });
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        navigate(Routes.SIMULATION);
      } catch (error) {
        console.error('Error:', error);
      }
    },
    [formData]
  );

  return (
    <div
      style={{
        position: 'absolute',
        top: '5%',
        left: 'calc(50% - 200px)',
        overflow: 'scroll',
      }}
    >
      {/* Card: https://www.radix-ui.com/themes/docs/components/card */}
      <Card
        style={{
          width: '400px',
        }}
      >
        <Heading as="h2" size="4" weight="bold" mb="4">
          Run a Simulation
        </Heading>
        <Link to={Routes.SIMULATION}>View previous simulation</Link>
        <Separator size="4" my="5" />
        <Form onSubmit={handleSubmit}>
          {/* 
            *********************************
            Body1
            *********************************
            */}
          <Heading as="h3" size="3" weight="bold">
            Body1
          </Heading>
          {/* Form: https://www.radix-ui.com/primitives/docs/components/form */}
          <FormField name="Body1.x">
            <FormLabel htmlFor="Body1.x">Initial X-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.x"
              name="Body1.x"
              value={formData.Body1.x}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.y">
            <FormLabel htmlFor="Body1.y">Initial Y-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.y"
              name="Body1.y"
              value={formData.Body1.y}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.z">
            <FormLabel htmlFor="Body1.z">Initial Z-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.z"
              name="Body1.z"
              value={formData.Body1.z}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.vx">
            <FormLabel htmlFor="Body1.vx">Initial X-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.vx"
              name="Body1.vx"
              value={formData.Body1.vx}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.vy">
            <FormLabel htmlFor="Body1.vy">Initial Y-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.vy"
              name="Body1.vy"
              value={formData.Body1.vy}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.vz">
            <FormLabel htmlFor="Body1.vz">Initial Z-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body1"
              name="Body1"
              value={formData.Body1.vz}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.mass">
            <FormLabel htmlFor="Body1.mass">Mass</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.mass"
              name="Body1.mass"
              value={formData.Body1.mass}
              onChange={handleChange}
              required
            />
          </FormField>
          {/* 
            *********************************
            Body2
            *********************************
             */}
          <Heading as="h3" size="3" weight="bold" mt="4">
            Body2
          </Heading>
          <FormField name="Body2.x">
            <FormLabel htmlFor="Body2.x">Initial X-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.x"
              name="Body2.x"
              value={formData.Body2.x}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.y">
            <FormLabel htmlFor="Body2.y">Initial Y-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.y"
              name="Body2.y"
              value={formData.Body2.y}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.z">
            <FormLabel htmlFor="Body2.z">Initial Z-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.z"
              name="Body2.z"
              value={formData.Body2.z}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.vx">
            <FormLabel htmlFor="Body2.vx">Initial X-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.vx"
              name="Body2.vx"
              value={formData.Body2.vx}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.vy">
            <FormLabel htmlFor="Body2.vy">Initial Y-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.vy"
              name="Body2.vy"
              value={formData.Body2.vy}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.vz">
            <FormLabel htmlFor="Body2.vz">Initial Z-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.vz"
              name="Body2.vz"
              value={formData.Body2.vz}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.mass">
            <FormLabel htmlFor="Body2.mass">Mass</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.mass"
              name="Body2.mass"
              value={formData.Body2.mass}
              onChange={handleChange}
              required
            />
          </FormField>
          <Flex justify="center" m="5">
            <Button type="submit">Submit</Button>
          </Flex>
        </Form>
      </Card>
    </div>
  );
};

export default SimulateForm;
